#include "game-server.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("GameServer");

LossEstimator::LossEstimator (Time window)
: m_window {window} {
  m_sendList.clear ();
  m_rtxList.clear ();
}

LossEstimator::~LossEstimator () {

}

void LossEstimator::SendUpdate (uint16_t num, Time now) {
  m_sendList.push_back (std::make_pair (num, now));
}

void LossEstimator::RtxUpdate (uint16_t num, Time now) {
  m_rtxList.push_back (std::make_pair (num, now));
}

double_t LossEstimator::GetLoss (Time now) {
  /* First clean up the history */
  while (!m_sendList.empty () && m_sendList.front ().second < now - m_window) {
    m_sendList.pop_front ();
  }
  while (!m_rtxList.empty () && m_rtxList.front ().second < now - m_window) {
    m_rtxList.pop_front ();
  }
  if (m_sendList.empty ()) {
    return m_rtxList.empty () ? 0 : 1;
  }
  else {
    double_t rtxSum = 0;
    double_t sendSum = 0;
    for (auto &rtx : m_rtxList) {
      rtxSum += rtx.first;
    }
    for (auto &send : m_sendList) {
      sendSum += send.first;
    }
    return rtxSum / sendSum;
  }
}

TypeId GameServer::GetTypeId() {
  static TypeId tid = TypeId ("ns3::GameServer")
    .SetParent<Application> ()
    .SetGroupName("spark-rtc")
    .AddConstructor<GameServer>()
  ;
  return tid;
};

GameServer::GameServer()
: m_sender{NULL}
, m_encoder{NULL}
, m_fecPolicy{NULL}
, m_rtxPolicy{""}
, m_socket{NULL}
, m_dataPktHistoryKey {}
, m_dataPktHistory {}
, m_frameDataPktCnt {}
, m_curRxHighestDataGlobalId{0}
, m_curRxHighestGlobalId{0}
, m_curContRxHighestGlobalId{0}
, m_isRecovery{false}
, m_nextFrameId{0}
, m_nextGroupId{0}
, m_nextBatchId{0}
, fps{0}
, bitrate{0}
, m_frameInterval{0}
, m_delayDdl{MilliSeconds(0)}
, pacing_flag{false}
, m_srcIP{}
, m_srcPort{0}
, m_destIP{}
, m_destPort{0}
, m_cc_enable{false}
, m_cc_algorithm{NOT_USE_CC}
, m_cc_timer{Timer::CANCEL_ON_DESTROY}
, m_cc_interval{Seconds(1)}
{};

GameServer::~GameServer () {};

void GameServer::Setup (
    Ipv4Address srcIP, uint16_t srcPort, Ipv4Address destIP, uint16_t destPort,
    uint8_t fps, Time delay_ddl, uint32_t bitrate, uint16_t interval,
    Ptr<FECPolicy> fecPolicy, std::string rtxPolicy, Time measure_window,
    uint16_t default_rtt, double_t default_bw,
    double_t default_loss, double_t default_group_delay,
    bool trace_set, std::string trace_file, Ptr<OutputStreamWrapper> fecStream,
    Ptr<OutputStreamWrapper> debugStream
  ) {
  // DEBUG("[GameServer] Server ip: " << srcIP << ", server port: " << srcPort << ", Client ip: " << destIP <<  ", dstPort: " << destPort);
  m_srcIP = srcIP;
  m_srcPort = srcPort;
  m_destIP = destIP;
  m_destPort = destPort;

  this->check_rtx_interval = MicroSeconds(1e3); // check for retransmission every 1ms

  m_fecPolicy = fecPolicy;
  m_rtxPolicy = rtxPolicy;
  this->pacing_flag = fecPolicy->GetPacingFlag ();
  // init frame index
  m_nextFrameId = 0;
  m_nextGroupId = 0;
  m_nextBatchId = 0;
  // init modules
  this->fps = fps;
  this->bitrate = bitrate;
  m_frameInterval = interval;
  m_encoder = Create<DumbVideoEncoder> (fps, bitrate, this, &GameServer::SendFrame);
  m_sender = Create<PacketSender> (this, m_frameInterval, delay_ddl, debugStream, 
    &GameServer::RcvACKPacket, &GameServer::RcvFrameAckPacket);
  m_delayDdl = delay_ddl;
  m_sender->SetNetworkStatistics (MilliSeconds (default_rtt), default_bw, default_loss, default_group_delay);
  this->send_group_cnt = 0;
  this->send_frame_cnt = 0;

  m_lossEstimator = Create<LossEstimator> (measure_window);

  m_maxPayloadSize = DataPacket (0, 0, 0).GetMaxPayloadSize ();

  m_fecStream = fecStream;
  m_debugStream = debugStream;
};

void GameServer::SetController(CC_ALG cc_algorithm) {
  switch (cc_algorithm)
  {
  case GCC:
    this->m_cc_enable = true;
    this->m_cc_algorithm = GCC;
    break;
  case NADA:
    this->m_cc_enable = true;
    this->m_cc_algorithm = NADA;
    break;
  case SCREAM:
    this->m_cc_enable = true;
    this->m_cc_algorithm = SCREAM;
    break;
  default:
    break;
  }
}

void GameServer::DoDispose() {};

void GameServer::StartApplication() {
  // init UDP socket
  this->InitSocket();
  m_sender->StartApplication (m_socket);
  if(this->m_cc_enable) {
    switch (this->m_cc_algorithm)
    {
    case GCC:
      m_sender->SetController (std::make_shared<rmcat::GccController> ());
      break;
    case NADA:
      m_sender->SetController (std::make_shared<rmcat::NadaController> ());
    default:
      break;
    }
    // this->m_cc_timer.SetFunction(&GameServer::UpdateBitrate, this);
    // this->m_cc_timer.SetDelay(this->m_cc_interval);
    // this->m_cc_timer.Schedule();
  }
  m_encoder->StartEncoding ();
  this->check_rtx_event = Simulator::Schedule(this->check_rtx_interval, &GameServer::CheckRetransmission, this);
};

void GameServer::StopApplication() {
  NS_LOG_ERROR("\n\n[Server] Stopping GameServer...");
  m_sender->GetBandwidthLossRate ();
  this->OutputStatistics();
  this->check_rtx_event.Cancel();
  m_socket->Close();
};

void GameServer::InitSocket() {
  if (m_socket == NULL) {
    m_socket = Socket::CreateSocket (GetNode (), UdpSocketFactory::GetTypeId ());
    auto res = m_socket->Bind (InetSocketAddress {m_srcIP, m_srcPort});
    NS_ASSERT (res == 0);
    m_socket->Connect (InetSocketAddress {m_destIP, m_destPort});
  }
};

void GameServer::StopEncoding() {
  m_encoder->StopEncoding();
};

uint32_t GameServer::GetNextGroupId() { return m_nextGroupId ++; };
uint32_t GameServer::GetNextBatchId() { return m_nextBatchId ++; };
uint32_t GameServer::GetNextFrameId() { return m_nextFrameId ++; };

TypeId GameServer::UnFECedPackets::GetTypeId() {
  static TypeId tid = TypeId ("ns3::GameServer::UnFECedPackets")
    .SetParent<Object> ()
    .SetGroupName("spark-rtc")
    .AddConstructor<GameServer::UnFECedPackets> ()
  ;
  return tid;
};

GameServer::UnFECedPackets::UnFECedPackets()
: param {}
, next_pkt_id_in_batch {0}
, next_pkt_id_in_group {0}
, pkts {}
{};

GameServer::UnFECedPackets::~UnFECedPackets() {};

void GameServer::UnFECedPackets::SetUnFECedPackets(
  FECPolicy::FECParam param,
  uint16_t next_pkt_id_in_batch, uint16_t next_pkt_id_in_group,
  std::vector<Ptr<DataPacket>> pkts
) {
  this->param = param;
  this->next_pkt_id_in_batch = next_pkt_id_in_batch;
  this->next_pkt_id_in_group = next_pkt_id_in_group;
  this->pkts = pkts;
};

void GameServer::CreatePacketBatch(
  std::vector<Ptr<VideoPacket>>& pkt_batch,
  FECPolicy::FECParam fec_param,
  bool new_group, uint32_t group_id, bool is_rtx) {

  // data packets
  std::vector<Ptr<DataPacket>> data_pkts;
  for(auto pkt : pkt_batch)
    data_pkts.push_back(DynamicCast<DataPacket, VideoPacket> (pkt));

  // FEC paramters
  double_t fec_rate =fec_param.fec_rate;

  // new_group cannot be true when new_batch is false
  NS_ASSERT(data_pkts.size() > 0);

  // if force_create_fec -> create FEC packets even though data packet num is less than fec_param
  uint16_t batch_data_num = data_pkts.size();
  /* since fec_rate is calculated by fec_count when tx_count == 0, use round to avoid float error. */
  uint16_t max_fec_num = UINT16_MAX;
  // *m_debugStream->GetStream () << "[Server] At " << Simulator::Now ().GetMilliSeconds () << 
  //   " quota is " << m_ccaQuotaPkt << " beforeFec " << round(batch_data_num * fec_rate) << std::endl;
  if (this->m_cc_enable) {
    max_fec_num = MAX (1, m_ccaQuotaPkt);
  }
  uint16_t batch_fec_num = MIN ((uint16_t) round(batch_data_num * fec_rate), max_fec_num);
  m_ccaQuotaPkt -= batch_fec_num;

  uint16_t group_data_num, group_fec_num;
  uint8_t tx_count = data_pkts.front()->GetTXCount();

  uint32_t batch_id = GetNextBatchId();
  uint16_t pkt_id_in_batch = 0, pkt_id_in_group = 0;
  uint32_t frameId = data_pkts.front ()->GetFrameId ();
  if (new_group) {
    group_id = GetNextGroupId();
    pkt_id_in_group = 0;
    group_data_num = batch_data_num;
    group_fec_num = batch_fec_num;
    if (std::find (m_frameIdToGroupId[frameId].begin (), m_frameIdToGroupId[frameId].end (), group_id) == m_frameIdToGroupId[frameId].end ()) 
      m_frameIdToGroupId[frameId].push_back (group_id);
  } else {
    group_data_num = data_pkts.front()->GetGroupDataNum();
    group_fec_num = data_pkts.front()->GetBatchFECNum();
  }
  // DEBUG("[Group INFO] gid " << group_id << ", group data num: " << group_data_num << ", group fec num: " << group_fec_num << ", batch id: " << batch_id << ", batch data num: " << batch_data_num << ", batch fec num: " << batch_fec_num);

  // to save some time
  pkt_batch.reserve(batch_data_num + batch_fec_num);

  // DEBUG("[GameServer] group id: " << group_id << ", batch_id: " << batch_id << ", batch_data_num: " << batch_data_num);
  // Assign batch & group ids for data packets
  for(auto data_pkt : data_pkts) {
    NS_ASSERT(data_pkt->GetPacketType() == PacketType::DATA_PKT);

    // Set batch info
    data_pkt->SetFECBatch(batch_id, batch_data_num, batch_fec_num, pkt_id_in_batch ++);

    // Set group info
    if (!new_group && data_pkt->GetGroupInfoSetFlag()) {
      // packets whose group info have already been set
      NS_ASSERT(data_pkt->GetGroupId() == group_id);
      continue;
    }
    // other packets
    data_pkt->SetFECGroup(group_id, group_data_num, group_fec_num, pkt_id_in_group ++);
  }

  // Generate FEC packets
  // create FEC packets and push into packet_group{
  for(uint16_t i = 0;i < batch_fec_num;i++) {
    Ptr<FECPacket> fec_pkt = Create<FECPacket> (tx_count, data_pkts);
    fec_pkt->SetFECBatch(batch_id, batch_data_num, batch_fec_num, pkt_id_in_batch ++);
    if(!is_rtx)
      fec_pkt->SetFECGroup(
        group_id, group_data_num, group_fec_num, pkt_id_in_group ++
      );
    else
      fec_pkt->SetFECGroup(
        group_id, group_data_num, group_fec_num, VideoPacket::RTX_FEC_GROUP_ID
      );
    fec_pkt->SetEncodeTime(Simulator::Now());
    pkt_batch.push_back(fec_pkt);
  }
  // DEBUG(pkt_batch.size() << ", "<< fec_param.fec_group_size << ", "<< batch_id << pkt_id_in_batch << group_id << pkt_id_in_group);
  return ;
};


void GameServer::CreateFirstPacketBatch(
  std::vector<Ptr<VideoPacket>>& pkt_batch, FECPolicy::FECParam fec_param) {
  NS_LOG_FUNCTION(pkt_batch.size());

  NS_ASSERT(pkt_batch.size() > 0);

  this->CreatePacketBatch(
    pkt_batch, fec_param, true /* new_group */, 0, false /* not rtx */
  );

  this->send_group_cnt ++;
};

void GameServer::CreateRTXPacketBatch(
  std::vector<Ptr<VideoPacket>>& pkt_batch, FECPolicy::FECParam fec_param) {
  NS_LOG_FUNCTION(pkt_batch.size());

  NS_ASSERT(pkt_batch.size() > 0);
  NS_ASSERT(pkt_batch.size() == fec_param.fec_group_size);

  uint32_t group_id = pkt_batch.front()->GetGroupId();

  this->CreatePacketBatch(
    pkt_batch, fec_param, false /* new_group */, group_id, true /* is rtx */
  );
};

Time GameServer::GetDispersion (Ptr<DataPacket> pkt) {
  uint32_t batchSize = pkt->GetBatchDataNum () + pkt->GetBatchFECNum ();
  auto statistic = m_sender->GetNetworkStatistics ();
  return std::min (batchSize * statistic->oneWayDispersion + MicroSeconds (500), MilliSeconds (m_frameInterval));
}

bool GameServer::IsRtxTimeout (Ptr<DataPacket> pkt, Time rto) {
  Time now = Simulator::Now ();
  Time enqueueTime = pkt->GetEnqueueTime ();
  Time lastSendTime = pkt->GetSendTime ();
  // Decide if the packet needs to be retransmitted
  auto statistic = m_sender->GetNetworkStatistics ();
  if (rto == Time (0)) {
    /* rto = max (avg + 4 * stdev, 2 * avg) 
       basically following the pto plan from 
       RFC 8985 - The RACK-TLP Loss Detection Algorithm for TCP
       https://datatracker.ietf.org/doc/rfc8985/
    */
    rto = Max (statistic->srtt + 4 * statistic->rttSd, 2 * statistic->srtt);
  }
  if (m_fecPolicy->GetFecName () == "HairpinPolicy" && pkt->GetTXCount () > 1) {
    rto = Max (statistic->srtt + 4 * statistic->rttSd, 1.5 * statistic->srtt);
  }
  rto += GetDispersion (pkt) + MicroSeconds (500);
  return (now > enqueueTime && (now - lastSendTime > rto));
};

bool GameServer::MissesDdl (Ptr<DataPacket> pkt) {
  Time now = Simulator::Now ();
  Time encodeTime = pkt->GetEncodeTime ();
  // Decide if it's gonna miss ddl. Could be more strict
  auto statistic = m_sender->GetNetworkStatistics ();
  // *m_debugStream->GetStream () << now.GetMilliSeconds () << 
  //   " encodeTime " << encodeTime.GetMilliSeconds () <<
  //   " MinRtt " << statistic->minRtt.GetMilliSeconds () <<
  //   " delayDdl " << m_delayDdl << 
  //   " missesDdl " << (now - encodeTime + statistic->minRtt / 2 > m_delayDdl) << std::endl; 
  return (now - encodeTime + statistic->minRtt / 2 > m_delayDdl);
};

void GameServer::StorePackets (std::vector<Ptr<VideoPacket>> pkts) {
  // store data packets in case of retransmission
  for (auto pkt : pkts) {
    if (pkt->GetPacketType() != PacketType::DATA_PKT)
      continue;
    Ptr<DataPacket> dataPkt = DynamicCast<DataPacket, VideoPacket> (pkt);
    Ptr<GroupPacketInfo> info = Create<GroupPacketInfo> (pkt->GetGroupId (), pkt->GetPktIdGroup (), 
      dataPkt->GetDataGlobalId (), dataPkt->GetGlobalId ());
    m_dataPktHistoryKey.push_back (info);
    m_dataPktHistory[info->m_groupId][info->m_pktIdInGroup] = dataPkt;
  }
};

void GameServer::SendPackets (std::deque<Ptr<DataPacket>> pkts, Time ddlLeft, uint32_t frameId, bool isRtx) {

  m_ccaQuotaPkt -= pkts.size ();

  std::vector<Ptr<VideoPacket>> pktToSendList;
  std::vector<Ptr<VideoPacket>> tmpList;
  FECPolicy::FECParam fecParam;

  if (m_frameDataPktCnt.find (frameId) == m_frameDataPktCnt.end()) {
    NS_ASSERT_MSG(false, "No frame size info");
  }
  uint8_t frameSize = m_frameDataPktCnt[frameId];
  // Not retransmission packets:
  // Group the packets as fec_param's optimal
  if (!isRtx) {
    // Get a FEC parameter in advance to divide packets into groups
    // Default
    m_frameIdToGroupId[frameId].clear ();
    fecParam = GetFECParam (pkts.size(), m_encoder->GetBitrate (), ddlLeft, false, isRtx, frameSize);
    NS_LOG_FUNCTION ("fecParam " << Simulator::Now ().GetMilliSeconds () << 
      " loss " << m_sender->GetNetworkStatistics ()->curLossRate <<
      " frameSize " << (int) frameSize <<
      " ddlLeft " << ddlLeft <<
      " rtt " << m_sender->GetNetworkStatistics ()->srtt.GetMilliSeconds () <<
      " dispersion " << m_sender->GetNetworkStatistics ()->oneWayDispersion.GetMilliSeconds () <<
      " blockSize " << fecParam.fec_group_size <<
      " bitRate " << m_encoder->GetBitrate () <<
      " fecRate " << fecParam.fec_rate);

    // divide packets into groups
    while (pkts.size() >= fecParam.fec_group_size) {
      tmpList.clear();

      // get fec_param.group_size data packets from pkts
      for (uint16_t i = 0; i < fecParam.fec_group_size; i++) {
        tmpList.push_back (pkts.front());
        pkts.pop_front ();
      }

      // group them into a pkt_batch
      CreateFirstPacketBatch (tmpList, fecParam);

      // insert them into sending queue
      pktToSendList.insert (pktToSendList.end (), tmpList.begin(), tmpList.end());
    }
  }
  if (!pkts.empty ()) {
  /* Retransmission packets and tail packets:
     Group the remaininng packets as a single group */
    tmpList.clear();

    // not pacing
    // send out all packets
    // get all data packets left from data_pkt_queue
    fecParam = GetFECParam (pkts.size(), m_encoder->GetBitrate(), ddlLeft, true, isRtx, frameSize);
    while (!pkts.empty ()) {
      tmpList.push_back (pkts.front());
      pkts.pop_front ();
    }
    // group them into a pkt_batch!
    if (isRtx) 
      CreateRTXPacketBatch (tmpList, fecParam); /* rtx packets */
    else 
      CreateFirstPacketBatch (tmpList, fecParam);   /* remaining tail packets that cannot form a full fec group */
    pktToSendList.insert (pktToSendList.end (), tmpList.begin(), tmpList.end());
  }
  
  NS_ASSERT (!pktToSendList.empty ());

  *m_fecStream->GetStream () << Simulator::Now ().GetMilliSeconds () <<
    " frameId " << frameId <<
    " groupId " << pktToSendList[0]->GetGroupId () <<
    " txCnt " << unsigned(pktToSendList[0]->GetTXCount ()) <<
    " batchId " << pktToSendList[0]->GetBatchId () <<
    " batchDataNum " << pktToSendList[0]->GetBatchDataNum () <<
    " batchFecNum " << pktToSendList[0]->GetBatchFECNum () <<
    " encodeTime " << pktToSendList[0]->GetEncodeTime ().GetMilliSeconds () << 
    " measuredRtt " << m_sender->GetNetworkStatistics ()->srtt.GetMilliSeconds () <<
    " measuredLoss " << m_sender->GetNetworkStatistics ()->curLossRate <<
    " measuredRttStd " << m_sender->GetNetworkStatistics ()->rttSd.GetMilliSeconds () <<
    " fecRate " << m_curFecParam.fec_rate <<
    std::endl;

  // Set enqueue time
  for (auto pkt : pktToSendList)
    pkt->SetEnqueueTime (Simulator::Now());
  // Send out packets
  if (isRtx) 
    m_sender->SendRtx (pktToSendList);
  else 
    m_sender->SendFrame (pktToSendList);
  // store data packets in case of retransmissions
  StorePackets (pktToSendList);
  m_lossEstimator->SendUpdate (pktToSendList.size (), Simulator::Now ());
}

void GameServer::SendFrame(uint8_t * buffer, uint32_t data_size) {

  UpdateBitrate ();
  double_t bitrate = m_encoder->GetBitrate ();
  data_size = (uint32_t) (bitrate * 1000.0 / 8 / this->fps);
  data_size = MAX(data_size, 200);
  if (!data_size)
    return;

  this->send_frame_cnt ++;

  /* 1. Create data packets of a frame */
  // calculate the num of data packets needed
  uint32_t frame_id = this->GetNextFrameId();
  uint16_t pkt_id = 0;
  uint16_t data_pkt_max_payload = DataPacket::GetMaxPayloadSize();
  uint16_t data_pkt_num = data_size / data_pkt_max_payload
     + (data_size % data_pkt_max_payload != 0);    /* ceiling division */

  std::deque<Ptr<DataPacket>> data_pkt_queue;

  // create packets
  for(uint32_t data_ptr = 0; data_ptr < data_size; data_ptr += data_pkt_max_payload) {
    Ptr<DataPacket> data_pkt = Create<DataPacket>(frame_id, data_pkt_num, pkt_id);
    data_pkt->SetEncodeTime(Simulator::Now());
    data_pkt->SetPayload(nullptr, MIN(data_pkt_max_payload, data_size - data_ptr));
    data_pkt_queue.push_back(data_pkt);
    pkt_id = pkt_id + 1;
  }
  // record the number of data packets in a single frame
  m_frameDataPktCnt[frame_id] = data_pkt_num;

  // DEBUG("[GameServer] frame pkt num: " << data_pkt_queue.size());
  NS_ASSERT(data_pkt_queue.size() > 0);
  /* End of 1. Create data packets of a frame */

  auto statistics = m_sender->GetNetworkStatistics ();
  SendPackets (
    data_pkt_queue,
    m_delayDdl - data_pkt_queue.size() * statistics->oneWayDispersion,
    frame_id,
    false
  );
}

void GameServer::RetransmitGroup (uint32_t groupId) {
  // cannot find the packets to send
  NS_ASSERT (m_dataPktHistory.find (groupId) != m_dataPktHistory.end ());

  int txCnt = -1;
  Time encodeTime = Time (0);
  Time now = Simulator::Now ();

  std::deque<Ptr<DataPacket>> dataPktRtxQueue;

  auto groupDataPkt = m_dataPktHistory[groupId];
  // find all packets that belong to the same group and retransmit them
  for (const auto& [pktId, dataPkt] : groupDataPkt) {
    if (txCnt == -1) 
      txCnt = dataPkt->GetTXCount () + 1;
    if (encodeTime == Time (0)) 
      encodeTime = dataPkt->GetEncodeTime ();

    dataPkt->SetTXCount (txCnt);
    dataPkt->SetEnqueueTime (now);
    dataPkt->ClearFECBatch ();
    dataPktRtxQueue.push_back (dataPkt);
  }
  for (auto it = m_dataPktHistoryKey.begin (); it != m_dataPktHistoryKey.end ();) {
    Ptr<GroupPacketInfo> info = *it;
    if (info->m_groupId == groupId)
      it = m_dataPktHistoryKey.erase (it);
    else
      it ++;
  }
  groupDataPkt.clear ();

  if (!dataPktRtxQueue.empty ()) {
    m_lossEstimator->RtxUpdate (dataPktRtxQueue.size (), Simulator::Now ());
    // all packets in the same group belong to the same frame
    uint32_t frameId = dataPktRtxQueue.front ()->GetFrameId ();
    SendPackets (dataPktRtxQueue, m_delayDdl - (now - encodeTime), frameId, true);
  }
}

void GameServer::CheckRetransmission () {
  Time now = Simulator::Now ();
  this->check_rtx_event = Simulator::Schedule (this->check_rtx_interval, 
    &GameServer::CheckRetransmission, this);
  auto statistic = m_sender->GetNetworkStatistics ();

  bool isFront = true;
  /* 1) check for packets that will definitely miss ddl */
  for (auto it = m_dataPktHistoryKey.begin (); it != m_dataPktHistoryKey.end ();) {
    Ptr<GroupPacketInfo> info = (*it);
    if (info->m_state != GroupPacketInfo::PacketState::RCVD_PREV_DATA) {
      // if we cannot find it in m_dataPktHistory, and it's not a fake hole (data rcvd)
      if (m_dataPktHistory.find (info->m_groupId) == m_dataPktHistory.end ()) {
        it = m_dataPktHistoryKey.erase (it);
        continue;
      }
      auto groupDataPkt = &m_dataPktHistory[info->m_groupId];
      if (groupDataPkt->find (info->m_pktIdInGroup) == groupDataPkt->end ()) {
        it = m_dataPktHistoryKey.erase (it);
        continue;
      }
    }

    // we can find it in m_dataPktHistory, check if it's timed out
    if (isFront && (info->m_state == GroupPacketInfo::PacketState::RCVD_PREV_DATA || 
        MissesDdl (m_dataPktHistory[info->m_groupId][info->m_pktIdInGroup]))) {
      /* only remove pkts from begin ()! otherwise will create holes 
         remove it in m_dataPktHistory */
      m_dataPktHistory[info->m_groupId].erase (info->m_pktIdInGroup);
      if (m_dataPktHistory[info->m_groupId].empty ())
        m_dataPktHistory.erase (info->m_groupId);
      it = m_dataPktHistoryKey.erase (it);
    } else {
      // it's not a FIFO queue -- rtx packets are put to the end
      // we need to check if packets behind the first non-timeout packet will timeout
      isFront = false;
      it ++;
    }
  }

  /* 2) check for packets that exceeds rtx timer, needs to be retransmitted */
  std::unordered_set<uint32_t> rtxGroupId;

  /* Check if there are delayed rtx that is exactly the time to retransmit */
  for (auto it = m_delayedRtxGroup.begin (); it != m_delayedRtxGroup.end ();) {
    uint32_t groupId = (*it).first;
    Time rtxTime = (*it).second;
    if (rtxTime < now) {
      /* if there are still packets in that group, retransmit them,
         otherwise, just erase the group id since packets must have been received. */
      if (m_dataPktHistory.find (groupId) != m_dataPktHistory.end ())
        rtxGroupId.insert ((*it).first);
      it = m_delayedRtxGroup.erase (it);
    }
    else
      it ++;
  }

  uint16_t lastDataGlobalId = m_curRxHighestDataGlobalId;  /* for dup-ack check */
  if (m_dataPktHistoryKey.empty ()) {
    return;
  }

  // *m_debugStream->GetStream () << "[CheckRetransmission] " << now.GetMilliSeconds () << " curRx " << m_curRxHighestGlobalId;
  // for (auto it = m_dataPktHistoryKey.begin (); it != m_dataPktHistoryKey.end (); it++) {
  //   *m_debugStream->GetStream () << " (" << (*it)->m_dataGlobalId << " " << (*it)->m_globalId << ")";
  // }
  // *m_debugStream->GetStream () << std::endl;

  bool isLoop = true;
  bool hasHole = false;    /* whether we have found the first rtx packet or not */
  for (auto it = m_dataPktHistoryKey.end () - 1; isLoop && !m_dataPktHistoryKey.empty (); ) {
    bool shouldRtx = false;
    Ptr<DataPacket> pkt;
    Ptr<GroupPacketInfo> info = (*it);
    if (it == m_dataPktHistoryKey.begin ()) {
      isLoop = false;
    }    
    /* this packet has actually been received, thus no longer exists in m_dataPktHistory 
       this must be checked before the intialization of pkt = m_dataPktHistory, 
       otherwise there will be nullptr inside. */
    if (info->m_state == GroupPacketInfo::PacketState::RCVD_PREV_DATA) {
      if (m_sender->GetNetworkStatistics ()->srtt > 2 * m_lastRtt && m_fecPolicy->GetFecName () == "HairpinPolicy") {
        m_delayedRtxGroup[info->m_groupId] = now + m_lastRtt;
      }
      goto continueLoop;
    }
    
    pkt = m_dataPktHistory[info->m_groupId][info->m_pktIdInGroup];

    /* this packet is too early to retransmit */
    if (now - pkt->GetEncodeTime () < statistic->minRtt)
      goto continueLoop;

    /* this group has just been retransmitted */
    if (rtxGroupId.find (pkt->GetGroupId ()) != rtxGroupId.end ())
      goto continueLoop;

    NS_ASSERT (pkt->GetDataGlobalId () == info->m_dataGlobalId);

    /* If the FEC policy is RTX, we use the dup-ack retransmission policy
      we do not count on the PTO policy. The rtx for fec packets is buggy, do not use it */
    if (!hasHole) {
      // *m_debugStream->GetStream () << "[HoleCheck] m_dataGlobalId " << info->m_dataGlobalId << 
      //   " lastDataGlobalId " << lastDataGlobalId <<
      //   " m_isRecovery " << m_isRecovery << " globalId " << info->m_globalId << 
      //   " m_curContRxHighestGlobalId " << m_curContRxHighestGlobalId << std::endl;
      if (Uint16Less (int (info->m_dataGlobalId) + 1, int (lastDataGlobalId))) {
        /* holes in data packets */
        hasHole = true;
        *m_debugStream->GetStream () << "Holes(dataGlobalId): " << lastDataGlobalId << " " << info->m_dataGlobalId << std::endl;
      } else if (m_isRecovery && Uint16Less (info->m_globalId, m_curContRxHighestGlobalId)) {
        /* dataGlobalId is continuous, but fec packets might be lost */
        hasHole = true;
        *m_debugStream->GetStream () << "Holes(globalId): " << m_curContRxHighestGlobalId << " " << info->m_globalId << std::endl;
      }
    }

    if (hasHole) {
      /* If we detect a packet loss, be patient till dispersion */
      m_isRecovery = false; /* already found a hole, continue to find the next hole */
      if (m_delayedRtxGroup.find (pkt->GetGroupId ()) == m_delayedRtxGroup.end ())
        m_delayedRtxGroup[pkt->GetGroupId ()] = now + GetDispersion (pkt);
      else
        m_delayedRtxGroup[pkt->GetGroupId ()] = Min (m_delayedRtxGroup[pkt->GetGroupId ()], now + GetDispersion (pkt));
      *m_debugStream->GetStream () << "Delayed retransmit group " << pkt->GetGroupId () << 
        " to " << m_delayedRtxGroup[pkt->GetGroupId ()].GetMicroSeconds () <<
        " with dispersion " << GetDispersion (pkt).GetMilliSeconds () << " ms" << std::endl;
    }
    
    if (m_rtxPolicy == "pto") {
      /* For PTO, we can find it in m_dataPktHistory, check if it's timed out
         if this packet has not arrived at client
         do not retransmit it and all the packets behind 
         Specifically, PTO first goes through dupack, and then pto. */
      shouldRtx = IsRtxTimeout (pkt, Time (0));
    } else {
      shouldRtx = IsRtxTimeout (pkt, Seconds (1));   /* TCP RTO mechanism */
    }

    if (shouldRtx) {
      /* group all packets that are of the same group into a batch 
         and send out, but we cannot retransmit the whole group here 
         since we have no idea whether other packets in the same group
         need retransmitting or not (the iteration order) */
      uint32_t groupId = pkt->GetGroupId ();
      rtxGroupId.insert (groupId);
    }

continueLoop:
    it --;
    lastDataGlobalId = info->m_dataGlobalId;
  }

  /* Retransmit all the lost packets found in this round 
     We can only retransmit in the end because we're now iterating 
     from the newest to the oldest */
  for (uint32_t groupId : rtxGroupId) {
    if (m_delayedRtxGroup.find (groupId) != m_delayedRtxGroup.end ()) {
      /* if this group is also waiting to be retransmitted, erase it
         since we are now retransmitting that group. */
      m_delayedRtxGroup.erase (groupId);
    }
    RetransmitGroup (groupId);
    *m_debugStream->GetStream () << "Retransmit group " << groupId << std::endl;
  }
  
  m_lastRtt = m_sender->GetNetworkStatistics ()->srtt;
};

/* Remove packet history records when we receive an ACK packet */
void GameServer::RcvACKPacket (Ptr<AckPacket> ackPkt) {
  std::vector<Ptr<GroupPacketInfo>> pktInfos = ackPkt->GetAckedPktInfos ();

  for (Ptr<GroupPacketInfo> pktInfo : pktInfos) {
   if (!m_isRecovery) {
      m_curContRxHighestGlobalId = pktInfo->m_globalId;
      if (Uint16Less (int (m_curRxHighestGlobalId) + 1, pktInfo->m_globalId)) {
        m_isRecovery = true;
      }
    }
    m_curRxHighestGlobalId = pktInfo->m_globalId;
    // find the packet in m_dataPktHistory
    if (m_dataPktHistory.find (pktInfo->m_groupId) != m_dataPktHistory.end ()) {
      auto groupDataPkt = &m_dataPktHistory[pktInfo->m_groupId];
      if (groupDataPkt->find (pktInfo->m_pktIdInGroup) != groupDataPkt->end ()) {
        // erase pkt_id_in_frame
        groupDataPkt->erase (pktInfo->m_pktIdInGroup);
      }
      // erase frame_id if neccesary
      if (m_dataPktHistory[pktInfo->m_groupId].empty ()) {
        m_dataPktHistory.erase (pktInfo->m_groupId);
      }
    }
    for (auto it = m_dataPktHistoryKey.begin (); it != m_dataPktHistoryKey.end ();) {
      Ptr<GroupPacketInfo> senderInfo = (*it);
      if (senderInfo->m_groupId == pktInfo->m_groupId && senderInfo->m_pktIdInGroup == pktInfo->m_pktIdInGroup) {
        if (senderInfo->m_globalId == pktInfo->m_globalId) {
          it = m_dataPktHistoryKey.erase (it);
          if (Uint16Less (m_curRxHighestDataGlobalId, senderInfo->m_dataGlobalId))
            m_curRxHighestDataGlobalId = senderInfo->m_dataGlobalId;
        } else
          senderInfo->m_state = GroupPacketInfo::PacketState::RCVD_PREV_DATA;
        break;
      } else {
        it ++;
      }
    }
  }
};

void GameServer::RcvFrameAckPacket (Ptr<FrameAckPacket> frameAckPkt) {
  uint32_t frameId = frameAckPkt->GetFrameId ();
  for (auto groupId : m_frameIdToGroupId[frameId]) {
    if (m_dataPktHistory.find (groupId) != m_dataPktHistory.end ()) {
      m_dataPktHistory.erase (groupId);
    }
    for (auto info : m_dataPktHistoryKey) {
      if (info->m_groupId == groupId) {
        /* we are not sure if this info is in the front of the queue: 
           blindly remove the info could lead to the incorrect judgement 
           of previous data packet */
        info->m_state = GroupPacketInfo::PacketState::RCVD_PREV_DATA;
      }
    }
  }
  m_frameIdToGroupId.erase (frameId);
};

Ptr<Socket> GameServer::GetSocket() {
  return m_socket;
};

FECPolicy::FECParam GameServer::GetFECParam (
    uint16_t maxGroupSize, uint32_t bitrate,
    Time ddlLeft, bool fixGroupSize,
    bool isRtx, uint8_t frameSize
  ) {
  m_sender->GetNetworkStatistics ()->curLossRate = m_lossEstimator->GetLoss (Simulator::Now ());
  auto fecParam = m_fecPolicy->GetFECParam (m_sender->GetNetworkStatistics (), bitrate, 
    m_delayDdl.GetMilliSeconds (), (uint16_t) floor (ddlLeft.GetMicroSeconds () / 1e3), 
    isRtx, frameSize, maxGroupSize, fixGroupSize);
  m_curFecParam = fecParam;
  return fecParam;
};


void GameServer::UpdateBitrate(){
  if(!this->m_cc_enable) return;
  m_sender->UpdateSendingRate ();
  double_t bitrate = m_sender->GetSendingRate ();
  m_ccaQuotaPkt = bitrate / 8.0 / this->fps / DataPacket::GetMaxPayloadSize ();
  if (m_ccaQuotaPkt > 50) {
    bitrate = bitrate * 50 / m_ccaQuotaPkt;
    m_ccaQuotaPkt = 50;
  }
  double last_goodput_ratio = m_sender->GetGoodputRatio ();
  NS_ASSERT_MSG(last_goodput_ratio<=1 , "Goodput ratio must not be greater than 1");

  double_t goodput_ratio = 0.5;
  bitrate *= goodput_ratio;

  m_encoder->SetBitrate((uint32_t)(bitrate / 1000.));
};

void GameServer::OutputStatistics() {
  NS_LOG_ERROR("[Server] Total Frames: " << this->send_frame_cnt);
  NS_LOG_ERROR("[Server] [Result] Total Frames: " << this->send_frame_cnt - 10);
  NS_LOG_ERROR("[Server] Total Groups: " << this->send_group_cnt);
}

}; // namespace ns3