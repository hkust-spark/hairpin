#include "game-client.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("GameClient");

TypeId GameClient::GetTypeId() {
    static TypeId tid = TypeId ("ns3::GameClient")
        .SetParent<Application> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<GameClient>()
    ;
    return tid;
};

GameClient::GameClient()
: receiver{NULL}
, decoder{NULL}
, m_socket{NULL}
, fps{0}
, delay_ddl{MilliSeconds(0)}
, rtt{MilliSeconds(20)}
, m_peerIP{}
, m_peerPort{0}
, m_localPort{0}
, m_receiver_window{32000}
, incomplete_groups {}
, complete_groups {}
, timeout_groups {}
{};

/*
GameClient::GameClient(uint8_t fps, Time delay_ddl) {
    this->InitSocket();
    this->receiver = Create<PacketReceiver> (this, &GameClient::ReceivePacket, this->m_socket);
    this->decoder = Create<VideoDecoder> ();
    this->fps = fps;
    this->delay_ddl = delay_ddl;
};*/

GameClient::~GameClient() {

};

void GameClient::Setup (Ipv4Address srcIP, uint16_t srcPort, uint16_t destPort, uint8_t fps, Time delay_ddl, 
    uint32_t wndsize, uint16_t rtt, Ptr<OutputStreamWrapper> appStream, Ptr<OutputStreamWrapper> debugStream) {
    this->m_peerIP = srcIP;
    this->m_peerPort = srcPort;
    this->m_localPort = destPort;
    this->decoder = Create<VideoDecoder> (delay_ddl, this, appStream, &GameClient::ReplyFrameACK);
    this->fps = fps;
    this->delay_ddl = delay_ddl;
    this->m_receiver_window = wndsize;
    this->default_rtt = MilliSeconds(rtt/2);
    this->rcvd_pkt_cnt = 0;
    this->rcvd_data_pkt_cnt = 0;
    this->rcvd_fec_pkt_cnt = 0; 
    m_debugStream = debugStream;   
};

void GameClient::DoDispose() {

};

void GameClient::StartApplication(void) {
    this->InitSocket();
    this->receiver = Create<PacketReceiver> (this, &GameClient::ReceivePacket, this->m_socket, this->m_receiver_window);
};

void GameClient::StopApplication(void) {
    NS_LOG_ERROR("\n[Client] Stopping GameClient...");
    this->OutputStatistics();
    this->m_socket->Close();
    this->receiver->StopRunning();
};

void GameClient::InitSocket() {
    if (this->m_socket == NULL) {
        this->m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        auto res = this->m_socket->Bind(InetSocketAddress{Ipv4Address::GetAny(),this->m_localPort});
        NS_ASSERT (res == 0);
        this->m_socket->Connect(InetSocketAddress{this->m_peerIP, this->m_peerPort});
    }
};

Ptr<Socket> GameClient::GetSocket() { return this->m_socket; };

void GameClient::ReplyACK(std::vector<Ptr<DataPacket>> data_pkts, uint16_t last_pkt_id) {
    DEBUG("[Client] At " << Simulator::Now().GetMilliSeconds() << " ms send ACK for packet " << last_pkt_id);
    std::vector<Ptr<GroupPacketInfo>> pkt_infos;
    for(auto data_pkt : data_pkts) {
        DEBUG("[Client] At " << Simulator::Now().GetMilliSeconds() << " ms send ACK for group " << data_pkt->GetGroupId() << " pkt " << data_pkt->GetPktIdGroup());
        pkt_infos.push_back (Create<GroupPacketInfo> (data_pkt->GetGroupId (), data_pkt->GetPktIdGroup (), 
            data_pkt->GetGroupDataNum (), data_pkt->GetGlobalId ()));
    }
    Ptr<AckPacket> pkt = Create<AckPacket>(pkt_infos, last_pkt_id);
    this->receiver->SendPacket(pkt);
};

void GameClient::ReplyFrameACK(uint32_t frame_id, Time frame_encode_time) {
    Ptr<FrameAckPacket> pkt = Create<FrameAckPacket>(frame_id, frame_encode_time);
    this->receiver->SendPacket(pkt);
};

void GameClient::ReceivePacket(Ptr<VideoPacket> pkt) {
    auto group_id = pkt->GetGroupId();
    Time rcv_time = pkt->GetRcvTime();

    /* DEBUG */
    // if(pkt->GetTXCount() > 0)
        DEBUG("[Client] At " << Simulator::Now().GetMilliSeconds() <<
            " ms pkt rcvd: Type: " << pkt->GetPacketType() <<
            ", Global ID: " << unsigned(pkt->GetGlobalId()) <<
            ", TX cnt: " << unsigned(pkt->GetTXCount()) <<
            ", Group id: " << pkt->GetGroupId() <<
            ", Pkt id group: " << pkt->GetPktIdGroup() <<
            ", Batch id: " << pkt->GetBatchId() <<
            ", Pkt id batch: " << pkt->GetPktIdBatch() <<
            ", Batch data num: " << pkt->GetBatchDataNum() <<
            ", Batch fec num: " << pkt->GetBatchFECNum() <<
            ", Encode Time: " << pkt->GetEncodeTime().GetMilliSeconds() <<
            ", Rcv Time: " << pkt->GetRcvTime().GetMilliSeconds());
        
    if(pkt->GetTXCount() == 0) {
        if(pkt->GetPacketType() == PacketType::FEC_PKT)
            NS_LOG_FUNCTION("FEC packet (0) packet received!");
        // else if(pkt->GetPacketType() == PacketType::DATA_PKT)
        //     NS_LOG_FUNCTION("Data packet (0) received!");
    } else {
        NS_LOG_FUNCTION("RTX packet received!");
        if(pkt->GetPacketType() == PacketType::FEC_PKT)
            NS_LOG_FUNCTION("RTX FEC packet received!");
        else if(pkt->GetPacketType() == PacketType::DUP_FEC_PKT)
            NS_LOG_FUNCTION("RTX Dup FEC packet received!");
        else if(pkt->GetPacketType() == PacketType::DATA_PKT)
            NS_LOG_FUNCTION("RTX Data packet received!");
    }
    /* DEBUG end */

    /* STATISTICS */
    // statistic: rtx count
    if(this->rcvd_pkt_rtx_count.find(pkt->GetTXCount()) == this->rcvd_pkt_rtx_count.end())
        this->rcvd_pkt_rtx_count[pkt->GetTXCount()] = 0;
    this->rcvd_pkt_rtx_count[pkt->GetTXCount()] ++;

    this->rcvd_pkt_cnt ++;
    // categorize different types of rtx count
    if(pkt->GetPacketType() == PacketType::DATA_PKT){
        this->rcvd_data_pkt_cnt ++;
        if(this->rcvd_datapkt_rtx_count.find(pkt->GetTXCount()) == this->rcvd_datapkt_rtx_count.end())
            this->rcvd_datapkt_rtx_count[pkt->GetTXCount()] = 0;
        this->rcvd_datapkt_rtx_count[pkt->GetTXCount()] ++;
    }

    if(pkt->GetPacketType() == PacketType::FEC_PKT || pkt->GetPacketType() == PacketType::DUP_FEC_PKT){
        this->rcvd_fec_pkt_cnt ++;
        if(this->rcvd_fecpkt_rtx_count.find(pkt->GetTXCount()) == this->rcvd_fecpkt_rtx_count.end())
            this->rcvd_fecpkt_rtx_count[pkt->GetTXCount()] = 0;
        this->rcvd_fecpkt_rtx_count[pkt->GetTXCount()] ++;
    }
    /* STATISTICS end */

    /* 1) necessity check */
    // do not proceed if the group is timed out or complete
    if(this->complete_groups.find(group_id) != this->complete_groups.end())
        return;
    if(this->timeout_groups.find(group_id) != this->timeout_groups.end())
        return;
    /* End of 1) necessity check */

    /* 2) Insert/Create packet group */
    if(this->incomplete_groups.find(group_id) == this->incomplete_groups.end()) {
        // if it's the first packet of the packet group
        this->incomplete_groups[group_id] = Create<PacketGroup> (
            group_id, pkt->GetGroupDataNum(), pkt->GetEncodeTime()
        );
    }
    // DEBUG(group_id);
    NS_ASSERT(this->incomplete_groups.find(group_id) != this->incomplete_groups.end());
    this->incomplete_groups[group_id]->AddPacket(pkt, this->receiver->Get_FECgroup_delay());
    /* End of 2) Insert/Create packet group */

    auto group = this->incomplete_groups[group_id];

    /* 3) Get decoded packets */
    // get decoded packets and send them to decoder
    auto decode_pkts = group->GetUndecodedPackets();
    this->decoder->DecodeDataPacket(decode_pkts);
    /* End of 3) Get decoded packets */

    /* 4) replay ACK packet */
    this->ReplyACK(decode_pkts, pkt->GetGlobalId());
    /* End of 4) replay ACK packet */

    /* 5) Check whether a group a complete */
    // check if all data packets in this group have been retrieved
    if(group->CheckComplete()) {
        // debug
        // DEBUG("[Complete Group] " <<
        //     "Group ID: " << group->GetGroupId() <<
        //     ", Total DATA: " << group->GetDataNum() <<
        //     ", Max TX: " << unsigned(group->GetMaxTxCount()) <<
        //     ", First TX data: " << group->GetFirstTxDataCount() <<
        //     ", First TX FEC: " << group->GetFirstTxFECCount() <<
        //     ", RTX data: " << group->GetRtxDataCount() <<
        //     ", RTX FEC: " << group->GetRtxFECCount() <<
        //     ", Frame id: " << decode_pkts[0]->GetFrameId() <<
        //     ", encode: " << pkt->GetEncodeTime().GetMilliSeconds() <<
        //     ", now" << Simulator::Now().GetMilliSeconds());
        // rtx count
        uint8_t max_tx = group->GetMaxTxCount();
        if(this->rcvd_group_rtx_count.find(max_tx) == this->rcvd_group_rtx_count.end())
            this->rcvd_group_rtx_count[max_tx] = 0;
        this->rcvd_group_rtx_count[max_tx] ++;
        // group delay (packet interval)
        Time avg_pkt_itvl = group->GetAvgPktInterval();
        if(avg_pkt_itvl > MicroSeconds(0) && max_tx == 0)
            this->receiver->Set_FECgroup_delay(avg_pkt_itvl);
        this->complete_groups[group_id] = group;
        this->incomplete_groups.erase(group_id);
    }
    /* End of 5) Check whether a group a complete */

    /* 6) Check all incomplete groups for timeouts */
    for(auto it = this->incomplete_groups.begin();it != this->incomplete_groups.end();) {
        auto i_group_id = it->second->GetGroupId();
        // Assume all packets comes from the same frame
        // DEBUG("Group ID: " << it->second->GetGroupId() <<
        //     "group rcv size: " << it->second->decoded_pkts.size());
        if (rcv_time > it->second->GetEncodeTime() + this->delay_ddl) {
            // debug
            DEBUG("[Client Timeout] " <<
                "At " << Simulator::Now().GetMilliSeconds() <<
                "ms, Group ID: " << it->second->GetGroupId() <<
                ", Total DATA: " << it->second->GetDataNum() <<
                ", Max TX: " << unsigned(it->second->GetMaxTxCount()) <<
                ", First TX data: " << it->second->GetFirstTxDataCount() <<
                ", First TX FEC: " << it->second->GetFirstTxFECCount() <<
                ", RTX data: " << it->second->GetRtxDataCount() <<
                ", RTX FEC: " << it->second->GetRtxFECCount() <<
                ", Last Rcv Time: " << it->second->GetLastRcvTimeTx0().GetMilliSeconds() <<
                ", Rcv time: " << rcv_time.GetMilliSeconds() <<
                ", Encode Time: " << it->second->GetEncodeTime().GetMilliSeconds() <<
                ", Delay ddl: " << this->delay_ddl.GetMilliSeconds());
            // the whole group is timed out
            this->timeout_groups[i_group_id] = it->second;
            it = this->incomplete_groups.erase(it);
            continue;
        }
        it ++;
    }
    /* End of 6) check all incomplete groups for timeouts */
};

void GameClient::OutputStatistics() {
    this->decoder->GetDDLMissRate();
    NS_LOG_ERROR("\n[Client] Max TX Count: " << rcvd_pkt_rtx_count.size() - 1);
    NS_LOG_ERROR("[Client] All Packets: " << this->rcvd_pkt_cnt);
    for(auto it = rcvd_pkt_rtx_count.begin();it != rcvd_pkt_rtx_count.end();it++)
        NS_LOG_ERROR("\tTX: " << unsigned(it->first) << ", packet count: " << it->second);
    NS_LOG_ERROR("[Client] Data Packet: " << this->rcvd_data_pkt_cnt);
    for(auto it = rcvd_datapkt_rtx_count.begin();it != rcvd_datapkt_rtx_count.end();it++)
        NS_LOG_ERROR("\tTX: " << unsigned(it->first) << ", packet count: " << it->second);
    NS_LOG_ERROR("[Client] FEC Packet: " << this->rcvd_fec_pkt_cnt);
    for(auto it = rcvd_fecpkt_rtx_count.begin();it != rcvd_fecpkt_rtx_count.end();it++)
        NS_LOG_ERROR("\tTX: " << unsigned(it->first) << ", packet count: " << it->second);

    NS_LOG_ERROR("\n[Client] Groups: Received: " <<  this->complete_groups.size() + this->incomplete_groups.size() + this->timeout_groups.size() << ", Complete: " << this->complete_groups.size() << ", Incomplete: " << this->incomplete_groups.size() << ", Timeout: " << this->timeout_groups.size());
    uint32_t group_count = this->complete_groups.size() + this->incomplete_groups.size() + this->timeout_groups.size();
    for(auto it = rcvd_group_rtx_count.begin();it != rcvd_group_rtx_count.end();it++) {
        NS_LOG_ERROR("\tTX: " << unsigned(it->first) << ", group count: " << it->second << ", ratio: " << ((double_t) it->second) / group_count * 100 << "%");
    }
};

}; // namespace ns3