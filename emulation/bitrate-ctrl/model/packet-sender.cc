#include "packet-sender.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("PacketSender");

TypeId PacketSender::GetTypeId() {
    static TypeId tid = TypeId ("ns3::PacketSender")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

PacketSender::PacketSender (
    GameServer * game_server, uint16_t interval,
    Time delay_ddl, Ptr<OutputStreamWrapper> debugStream,
    void (GameServer::*ReportACKFunc)(Ptr<AckPacket>),
    void (GameServer::*ReportFrameAckFunc)(Ptr<FrameAckPacket>)
)
: game_server {game_server}
, ReportACKFunc {ReportACKFunc}
, ReportFrameAckFunc {ReportFrameAckFunc}
, m_netStat {NULL}
, m_queue {}
, pktsHistory {}
, m_send_wnd {}
, m_send_wnd_size {10000}
, m_goodput_wnd {}
, m_goodput_wnd_size {50000} // 32 ms
, goodput_pkts_inwnd {0}
, total_pkts_inwnd {0}
, m_goodput_ratio {1.}
, m_cc_enable {false}
, m_bitrate {0.}
, m_group_id {0}
, m_groupstart_TxTime {0}
, m_group_size {0}
, m_prev_id {0}
, m_prev_RxTime {0}
, m_prev_groupend_id {0}
, m_prev_groupend_RxTime {0}
, m_prev_group_size {0}
, m_prev_pkts_in_frame{0}
, m_curr_pkts_in_frame{0}
, m_prev_frame_TxTime{0}
, m_curr_frame_TxTime{0}
, m_prev_frame_RxTime{0}
, m_curr_frame_RxTime{0}
, m_interval{interval}
, inter_arrival{0}
, inter_departure{0}
, inter_delay_var{0}
, inter_group_size{0}
, m_firstFeedback {true}
, m_controller {NULL}
, m_pacing {false}
, m_pacing_interval {MilliSeconds(0)}
, m_pacingTimer {Timer::CANCEL_ON_DESTROY}
, m_send_time_limit {MilliSeconds(16)}
, m_sendevent {}
, m_netGlobalId {0}
, m_dataGlobalId {0}
, m_last_acked_global_id {0}
, m_delay_ddl {delay_ddl}
, m_finished_frame_cnt {0}
, m_timeout_frame_cnt {0}
, trace_set {false}
, m_debugStream {debugStream}
{
    NS_LOG_ERROR("[Sender] Delay DDL is: " << this->m_delay_ddl.GetMilliSeconds() << " ms");
    this->init_data_pkt_count = 0;
    this->other_pkt_count = 0;
    this->init_data_pkt_size = 0;
    this->other_pkt_size = 0;
    this->exclude_head_init_data_pkt_count = 0;
    this->exclude_head_other_pkt_count = 0;
    this->exclude_head_init_data_pkt_size = 0;
    this->exclude_head_other_pkt_size = 0;
};

PacketSender::~PacketSender() {};

void PacketSender::SetNetworkStatistics (
    Time defaultRtt, double_t bw, double_t loss, double_t delay
) {
    if (m_netStat == NULL)
        m_netStat = Create<FECPolicy::NetStat>();
    // initialize netstat
    m_netStat->curRtt           = defaultRtt;
    m_netStat->srtt             = defaultRtt;
    m_netStat->minRtt           = defaultRtt;
    m_netStat->rttSd            = Time (0);
    m_netStat->curBw            = bw;
    m_netStat->curLossRate      = loss;
    m_netStat->oneWayDispersion = MicroSeconds ((uint64_t) (delay * 1e3));
    // randomly generate loss seq
    m_netStat->loss_seq.clear();
    int next_seq = 0;
    for(uint16_t i = 0;i < bw * 1e6 / 8 * 0.1 / 1300/* packet count in 100ms */;i++) {
        bool lost = bool(std::rand() % 2);
        if(lost) {
            if(next_seq <= 0) {
                next_seq --;
            } else {
                m_netStat->loss_seq.push_back(next_seq);
                next_seq = 0;
            }
        } else {    // not lost
            if(next_seq >= 0) {
                next_seq ++;
            } else {
                m_netStat->loss_seq.push_back(next_seq);
                next_seq = 0;
            }
        }
    }
};

void PacketSender::UpdateRTT (Time rtt) {
    if (m_netStat->srtt == Time (0))
        m_netStat->srtt = rtt;
    // EWMA formulas are implemented as suggested in
    // Jacobson/Karels paper appendix A.2
    
    double m_alpha = 0.125, m_beta = 0.25;
    // SRTT <- (1 - alpha) * SRTT + alpha *  R'
    Time rttErr (rtt - m_netStat->srtt);
    double gErr = rttErr.ToDouble (Time::S) * m_alpha;
    m_netStat->srtt += Time::FromDouble (gErr, Time::S);

    // RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
    Time rttDifference = Abs (rttErr) - m_netStat->rttSd;
    m_netStat->rttSd += rttDifference * m_beta;

    m_netStat->curRtt = rtt;
    m_netStat->minRtt = Min (m_netStat->minRtt, rtt);
}

void PacketSender::SetNetworkStatisticsBytrace(uint16_t rtt /* in ms */, 
                                double_t bw/* in Mbps */,
                                double_t loss_rate)
{
    UpdateRTT (MilliSeconds (rtt));
    m_netStat->curBw = bw;
    m_netStat->curLossRate = loss_rate;
    if (m_controller)
        m_controller->UpdateLossRate (uint8_t (loss_rate * 256));
    
};

void PacketSender::StartApplication(Ptr<Socket> socket) {
    this->m_socket = socket;
    this->m_socket->SetRecvCallback(MakeCallback(&PacketSender::OnSocketRecv_sender,this));
    this->m_pacingTimer.SetFunction(&PacketSender::SendPacket,this);
    this->m_pacingTimer.SetDelay(this->m_pacing_interval);
    if(this->trace_set){
        this->UpdateNetstateByTrace();
    }
};

void PacketSender::SetController(std::shared_ptr<rmcat::SenderBasedController> controller) {
    this->m_cc_enable = true;
    this->m_controller = controller;
}

void PacketSender::SendFrame (std::vector<Ptr<VideoPacket>> packets)
{
    this->UpdateGoodputRatio();
    NS_LOG_FUNCTION("At time " << Simulator::Now().GetMilliSeconds() << ", " << packets.size() << " packets are enqueued");
    Ptr<PacketFrame> newFrame = Create<PacketFrame>(packets,false);
    newFrame->Frame_encode_time_ = packets[0]->GetEncodeTime();
    this->m_queue.push_back(newFrame);
    this->Calculate_pacing_rate();
    if(m_pacing){
        if(this->m_pacingTimer.IsExpired()){
            this->m_pacingTimer.Schedule();
        }
        else {
            this->m_pacingTimer.Cancel();
            this->m_pacingTimer.Schedule();
        }
    }
    else {
        SendPacket ();
    }

};

void PacketSender::SendRtx (std::vector<Ptr<VideoPacket>> packets)
{
    NS_LOG_FUNCTION("At time " << Simulator::Now().GetMilliSeconds() << ", " << packets.size() << " RTX packets are enqueued");
    Ptr<PacketFrame> newFrame = Create<PacketFrame>(packets,true);
    newFrame->Frame_encode_time_ = packets[0]->GetEncodeTime();
    this->m_queue.insert(this->m_queue.begin(),newFrame);
    this->Calculate_pacing_rate();
    if(m_pacing){
        if(this->m_pacingTimer.IsExpired()){
           this->m_pacingTimer.Schedule();
        }
    }
    else {
        SendPacket ();
    }
};

void PacketSender::Calculate_pacing_rate()
{
    Time time_now = Simulator::Now();
    uint32_t num_packets_left = 0;
    for(uint32_t i=0;i<this->num_frame_in_queue();i++) {
        if(this->m_queue[i]->retransmission) {
            continue;
        }
        num_packets_left = num_packets_left + this->m_queue[i]->Frame_size_in_packet();
        if(num_packets_left < 1) {
            NS_LOG_ERROR("Number of packet should not be zero.");
        }
        Time time_before_ddl_left = this->m_send_time_limit + this->m_queue[i]->Frame_encode_time_ - time_now;
        Time interval_max = time_before_ddl_left / num_packets_left;
        if(time_before_ddl_left <= Time(0)) {
            // TODO: DDL miss appears certain, need to require new IDR Frame from GameServer?
            NS_LOG_ERROR("DDL miss appears certain.");
        }
        else {
            if(interval_max < this->m_pacing_interval) {
                this->m_pacing_interval = interval_max;
                this->m_pacingTimer.SetDelay(interval_max);
            }
        }
    }
};

void PacketSender::OnSocketRecv_sender(Ptr<Socket> socket)
{
    Ptr<Packet> pkt = socket->Recv();
    Ptr<NetworkPacket> packet = NetworkPacket::ToInstance(pkt);
    auto pkt_type = packet->GetPacketType();
    Time now = Simulator::Now();

    if(pkt_type == ACK_PKT)
    {
        NS_LOG_FUNCTION("ACK packet received!");
        Ptr<AckPacket> ack_pkt = DynamicCast<AckPacket, NetworkPacket> (packet);
        // hand to GameServer for retransmission
        ((this->game_server)->*ReportACKFunc)(ack_pkt);

        // Update RTT and inter-packet delay
        uint16_t pkt_id = ack_pkt->GetLastPktId();
        DEBUG("[Sender] At " << Simulator::Now().GetMilliSeconds() << " ms rcvd ACK for packet " << pkt_id);
        if(this->pktsHistory.find(pkt_id) != this->pktsHistory.end()) {
            // RTT
            Ptr<SentPacketInfo> current_pkt = this->pktsHistory[pkt_id];
            current_pkt->pkt_ack_time = now;
            Time rtt = now - current_pkt->pkt_send_time;
            if(pkt_id >= this->m_last_acked_global_id) {
                if(!this->trace_set) this->UpdateRTT(rtt);
            }
            
            this->m_last_acked_global_id = pkt_id;

            // inter-packet delay
            if(this->pktsHistory.find(pkt_id - 1) != this->pktsHistory.end()) {
                Ptr<SentPacketInfo> last_pkt = this->pktsHistory[pkt_id - 1];
                if(last_pkt->batch_id == current_pkt->batch_id
                    && last_pkt->pkt_send_time == current_pkt->pkt_send_time
                    && last_pkt->pkt_ack_time != MicroSeconds(0)
                    && now >= last_pkt->pkt_ack_time) {
                    Time inter_pkt_delay = now - last_pkt->pkt_ack_time;
                    if(m_netStat->rt_dispersion == MicroSeconds(0))
                        m_netStat->rt_dispersion = inter_pkt_delay;
                    else m_netStat->rt_dispersion =
                        0.2 * inter_pkt_delay + 0.8 * m_netStat->rt_dispersion;
                }
            }
        }
        return;
    }

    if(pkt_type == FRAME_ACK_PKT) {
        NS_LOG_FUNCTION ("Frame ACK packet received!");
        Ptr<FrameAckPacket> frame_ack_pkt = DynamicCast<FrameAckPacket, NetworkPacket> (packet);
        ((this->game_server)->*ReportFrameAckFunc)(frame_ack_pkt);
        uint32_t frame_id = frame_ack_pkt->GetFrameId();
        Time frame_encode_time = frame_ack_pkt->GetFrameEncodeTime();
        if(now - frame_encode_time <= m_delay_ddl + MilliSeconds(1)) {
            this->m_finished_frame_cnt ++;
            NS_LOG_FUNCTION ("[Sender] At " << Simulator::Now().GetMilliSeconds() << " Frame ID: " << frame_id << " rcvs within ddls");
        } else {
            NS_LOG_FUNCTION ("[Sender Timeout Frame] At " << Simulator::Now().GetMilliSeconds() << " Frame ID: " << frame_id << " misses ddl, delay: " << (now - frame_encode_time).GetMilliSeconds() << " ms\n");
            this->m_timeout_frame_cnt ++;
        }
        return;
    }

    // netstate packet
    NS_ASSERT_MSG(pkt_type == NETSTATE_PKT, "Sender should receive FRAME_ACK_PKT, ACK_PKT or NETSTATE_PKT");
    Ptr<NetStatePacket> netstate_pkt = DynamicCast<NetStatePacket, NetworkPacket> (packet);
    auto states = netstate_pkt->GetNetStates();
    if(!this->trace_set){
        m_netStat->curBw = ((double_t)states->throughput_kbps) / 1000.;
        m_netStat->curLossRate = states->loss_rate;
        if(this->m_controller)
            this->m_controller->UpdateLossRate(uint8_t (states->loss_rate * 256));
    }
    m_netStat->loss_seq = states->loss_seq;
    m_netStat->oneWayDispersion = MicroSeconds(states->fec_group_delay_us);

    DEBUG("[Sender] At " << Simulator::Now().GetMilliSeconds() << " ms  bw = " << m_netStat->curBw << " Loss = " << m_netStat->curLossRate);

    // RTT is now estimated using ACK packets instead of NetState packet
    /*
    uint32_t owd_us = 0;
    if(!this->trace_set){
        if(states->recvtime_hist.size() < 1) {
            NS_LOG_FUNCTION("No received packet feedback! RTT not updated.");
        }
        else
        {
            NS_ASSERT(states->recvtime_hist.size() != 0);
            for(auto recvtime_item : states->recvtime_hist) {
                NS_ASSERT_MSG(this->pktsHistory.find(recvtime_item->pkt_id) != this->pktsHistory.end(), "Feedbacked packet must have been sent.");
                uint32_t st_us = this->pktsHistory[recvtime_item->pkt_id]->pkt_send_time;
                owd_us += recvtime_item->rt_us - st_us;
                //NS_LOG_DEBUG("pkt ID="<<recvtime_item->pkt_id<<", TxTime="<<st_us<<", RxTime="<<recvtime_item->rt_us<<", owd(us)="<<owd_us);
            }
            m_netStat->current_rtt = owd_us * 2 / states->recvtime_hist.size() / 1000;
            NS_LOG_FUNCTION("new estimated rtt: " << m_netStat->current_rtt << "ms");
        }
    }
    */

    if(this->m_cc_enable){
        uint64_t now_us = Simulator::Now().GetMicroSeconds();

        for(auto recvtime_item : states->recvtime_hist){
            uint16_t id = recvtime_item->pkt_id;
            uint64_t RxTime = recvtime_item->rt_us;
            uint64_t TxTime = m_controller->GetPacketTxTimestamp(id);
            // NS_ASSERT_MSG((RxTime <= now_us), "Receiving event and feedback event should be time-ordered.");
            if(this->m_firstFeedback) {
                this->m_prev_id = id;
                this->m_prev_RxTime = RxTime;
                this->m_group_id = 0;
                this->m_group_size = m_controller->GetPacketSize(id);
                this->m_groupstart_TxTime = TxTime;
                this->m_firstFeedback = false;
                this->m_curr_pkts_in_frame = 1;
                continue;
            }
            
            if((Uint64Less (this->m_groupstart_TxTime, TxTime) && TxTime - this->m_groupstart_TxTime < 6 * this->m_interval * 1000)|| 
               this->m_groupstart_TxTime == TxTime){
                // std::cout<<"group start:"<<m_groupstart_TxTime<<", tx:" <<TxTime<<std::endl;
                if((TxTime - this->m_groupstart_TxTime) > 10000) {
                    //Switching to another burst (named as group)
                    // update inter arrival and inter departure
                    if(this->m_group_id > 0){
                        NS_ASSERT_MSG(this->m_prev_pkts_in_frame>0 && this->m_prev_pkts_in_frame>0,"Consecutive frame must have pkts!");
                        inter_arrival = this->m_curr_frame_RxTime / this->m_curr_pkts_in_frame - this->m_prev_frame_RxTime / this->m_prev_pkts_in_frame;
                        //inter_departure = this->m_controller->UpdateDepartureTime(this->m_prev_groupend_id, this->m_prev_id);
                        inter_departure = this->m_curr_frame_TxTime / this->m_curr_pkts_in_frame - this->m_prev_frame_TxTime / this->m_prev_pkts_in_frame;
                        // if(inter_departure > 4e6){
                        //     std::cout<< "Group id: " << m_group_id
                        //             << ", inter_departure: " << inter_departure
                        //             << ", curr_frame_tx: " << m_curr_frame_TxTime
                        //             << ", prev_frame_tx: " << m_prev_frame_TxTime
                        //             << ", inter_arrival: " << inter_arrival
                        //             << ", curr_frame_rx: " << m_curr_frame_RxTime
                        //             << ", prev_frame_rx: " << m_prev_frame_RxTime
                        //             << ", curr_pkt_num: " << m_curr_pkts_in_frame
                        //             << ", prev_pkt_num: " << m_prev_pkts_in_frame
                        //             << std::endl;
                        // }
                        inter_delay_var = inter_arrival - inter_departure;
                        inter_group_size = this->m_group_size - this->m_prev_group_size;
                        //std::cout<<"inter_arrival "<<inter_arrival <<" inter_dep "<<inter_departure<<std::endl;
                        this->m_controller->processFeedback(now_us, id, RxTime, inter_arrival, inter_departure, inter_delay_var, inter_group_size, this->m_prev_RxTime);
                    }


                    // update group information
                    this->m_controller->PrunTransitHistory(this->m_prev_groupend_id);
                    this->m_prev_group_size = this->m_group_size;
                    this->m_prev_groupend_id = this->m_prev_id;
                    this->m_prev_groupend_RxTime = this->m_prev_RxTime;
                    this->m_group_id += 1;
                    this->m_group_size = 0;
                    this->m_groupstart_TxTime = TxTime;
                    this->m_prev_frame_TxTime = this->m_curr_frame_TxTime;
                    this->m_prev_frame_RxTime = this->m_curr_frame_RxTime;
                    this->m_prev_pkts_in_frame = this->m_curr_pkts_in_frame;
                    this->m_curr_frame_TxTime = 0;
                    this->m_curr_frame_RxTime = 0;
                    this->m_curr_pkts_in_frame = 0;
                }

                this->m_curr_pkts_in_frame += 1;
                this->m_curr_frame_TxTime += TxTime;
                this->m_curr_frame_RxTime += RxTime;       

                this->m_group_size += this->m_controller->GetPacketSize(id);
                this->m_prev_id = id;
                this->m_prev_RxTime = RxTime;
            }
            else{
                // std::cout<<"group id: "<< this->m_group_id
                //          <<", group start tx: " << this->m_groupstart_TxTime
                //          <<", tx: " << TxTime << std::endl;
            }
            
        }
    }
};


Ptr<FECPolicy::NetStat> PacketSender::GetNetworkStatistics() { return m_netStat; };

double_t PacketSender::GetBandwidthLossRate() {
    double_t bandwidth_loss_rate_count, bandwidth_loss_rate_size;
    if(this->init_data_pkt_count + this->other_pkt_count == 0)
        return 0;
    bandwidth_loss_rate_count =
        ((double_t) this->other_pkt_count) /
        (this->init_data_pkt_count);
    bandwidth_loss_rate_size =
        ((double_t) this->other_pkt_size) /
        (this->init_data_pkt_size);
    NS_LOG_ERROR("[Sender] Initial data packets: " << this->init_data_pkt_count << ", other packets:" << this->other_pkt_count);
    NS_LOG_ERROR("[Sender] Bandwidth loss rate: " << bandwidth_loss_rate_count * 100 << "% (count), " << bandwidth_loss_rate_size * 100 << "% (size)");

    if(this->exclude_head_init_data_pkt_count > 0) {
        double_t exclude_head_bandwidth_loss_rate_count, exclude_head_bandwidth_loss_rate_size;
        exclude_head_bandwidth_loss_rate_count =
            ((double_t) this->exclude_head_other_pkt_count) /
            (this->exclude_head_init_data_pkt_count);
        exclude_head_bandwidth_loss_rate_size =
            ((double_t) this->exclude_head_other_pkt_size) /
            (this->exclude_head_init_data_pkt_size);
        NS_LOG_ERROR("[Sender] [Result] Initial data packets: " << this->exclude_head_init_data_pkt_count << ", other packets:" << this->exclude_head_other_pkt_count);
        NS_LOG_ERROR("[Sender] [Result] Bandwidth loss rate: " << exclude_head_bandwidth_loss_rate_count * 100 << "% (count), " << exclude_head_bandwidth_loss_rate_size * 100 << "% (size)");
    }
    NS_LOG_ERROR("[Sender] Played frames: " << this->m_finished_frame_cnt << ", timeout frames: " << this->m_timeout_frame_cnt);

    // log bandwidth redundancy result to file
    //std::ofstream f_bw("results/bw_redundant.txt", std::ios::app);
    //if(!f_bw){
    //    DEBUG("File (bw_redundant.txt) open error"<<std::endl;
    //}
    //else{
    //    f_bw << bandwidth_loss_rate_count << " " << bandwidth_loss_rate_size);
    //    f_bw.close();
    //}

    return bandwidth_loss_rate_count;
};

void PacketSender::UpdateSendingRate(){
    if(this->m_cc_enable){
        this->m_bitrate = this->m_controller->getSendBps();
    }
};

double PacketSender::GetSendingRate(){
    return this->m_bitrate;
};

double PacketSender::GetGoodputRatio(){
    return this->m_goodput_ratio;
};

void PacketSender::UpdateGoodputRatio(){
    if(this->total_pkts_inwnd > 1 && this->goodput_pkts_inwnd > 0) {
            this->m_goodput_ratio = (double)(this->goodput_pkts_inwnd) / (double)(this->total_pkts_inwnd);
            // if(m_goodput_ratio==0){
                DEBUG("[Sender] goodput size: "<<goodput_pkts_inwnd<<", total:"<<total_pkts_inwnd<<"\n");
            // }
        }
};

void PacketSender::UpdateNetstateByTrace()
{
    // Load trace file
    std::ifstream trace_file;
    trace_file.open(this->trace_filename);

    // Initialize storing variables
    std::string trace_line;
    std::vector<std::string> trace_data;
    std::vector<std::string> bw_value;
    std::vector<std::string> rtt_value;
    uint16_t rtt;
    double_t bw;
    double_t lr;

    uint64_t past_time = 0;

    // Set netstate for every tracefile line
    while (std::getline(trace_file,trace_line))
    {   
        rtt_value.clear();
        bw_value.clear();
        trace_data.clear();
        SplitString(trace_line, trace_data," ");
        SplitString(trace_data[0], bw_value, "Mbps");
        SplitString(trace_data[1], rtt_value, "ms");
        rtt = (uint16_t) std::atof(rtt_value[0].c_str());
        bw = std::atof(bw_value[0].c_str());
        lr = std::atof(trace_data[2].c_str());
        this->m_settrace_event = Simulator::Schedule(
            MilliSeconds(rtt + past_time),&PacketSender::SetNetworkStatisticsBytrace, 
            this, rtt, bw, lr);
        past_time += this->m_interval;
    }
    
};

void PacketSender::SetTrace(std::string tracefile)
{
    this->trace_set = true;
    this->trace_filename = tracefile;
};

void PacketSender::SetTimeLimit (const Time& limit)
{
    this->m_send_time_limit = limit;
};

void PacketSender::Pause()
{
    if(m_pacing)
    {
        this->m_pacingTimer.Cancel();
    }
    else {
        Simulator::Cancel(this->m_sendevent);
    }
};

void PacketSender::Resume()
{
    if(m_pacing)
    {
        this->m_pacingTimer.Schedule();
    }
    else
    {
        this->m_sendevent = Simulator::ScheduleNow(&PacketSender::SendPacket,this);
    }
};

void PacketSender::SendPacket()
{
    Time time_now = Simulator::Now();
    uint64_t NowUs = time_now.GetMicroSeconds();
    if(this->num_frame_in_queue() > 0){
        std::vector<Ptr<VideoPacket>>* current_frame = &this->m_queue[0]->packets_in_Frame;
        std::vector<Ptr<VideoPacket>>::iterator firstPkt = current_frame->begin();
        Ptr<VideoPacket> netPktToSend = *firstPkt;
        PacketType pktType = netPktToSend->GetPacketType ();
        netPktToSend->SetSendTime (time_now);
        netPktToSend->SetGlobalId (m_netGlobalId);
        if (pktType == PacketType::DATA_PKT) {
            Ptr<DataPacket> dataPkt = DynamicCast<DataPacket, VideoPacket> (netPktToSend);
            dataPkt->SetDataGlobalId (m_dataGlobalId);
            m_dataGlobalId = (m_dataGlobalId + 1) % 65536;
        }
        bool is_goodput = (pktType == PacketType::DATA_PKT)
                            && (netPktToSend->GetTXCount() == 0);

        Ptr<Packet> pktToSend = netPktToSend->ToNetPacket ();
        uint16_t pkt_size = pktToSend->GetSize();
        Ptr<SentPacketInfo> pkt_info = Create<SentPacketInfo>(m_netGlobalId,
                    netPktToSend->GetBatchId(),
                    time_now, pktType, is_goodput, pkt_size);

        if(this->m_cc_enable) {
            // handle pkt information to cc controller
            this->m_controller->processSendPacket(NowUs, m_netGlobalId, pktToSend->GetSize());
        }

        // statistics
        if(is_goodput) {
            this->init_data_pkt_count ++;
            this->init_data_pkt_size += pkt_size;
            this->goodput_pkts_inwnd += pkt_size;
        } else {
            this->other_pkt_count ++;
            this->other_pkt_size += pkt_size;
        }
        this->total_pkts_inwnd += pkt_size;
        // DEBUG("[Sent packet] " <<
        //     "Packet Type: " << netPktToSend->GetPacketType() <<
        //     ", TX Count: " << unsigned(netPktToSend->GetTXCount()) <<
        //     ", Group id: " << netPktToSend->GetGroupId() <<
        //     ", Pkt id group: " << netPktToSend->GetPktIdGroup() <<
        //     ", Batch id: " << netPktToSend->GetBatchId() <<
        //     ", Pkt id batch: " << netPktToSend->GetPktIdBatch() <<
        //     ", Batch data num: " << netPktToSend->GetBatchDataNum() <<
        //     ", Batch fec num: " << netPktToSend->GetBatchFECNum() <<
        //     ", Encode Time: " << netPktToSend->GetEncodeTime().GetMilliSeconds());

        DEBUG("[Sender] At " << Simulator::Now().GetMilliSeconds() << " Send packet " << netPktToSend->GetGlobalId() << ", Group id: " << netPktToSend->GetGroupId());
        m_socket->Send(pktToSend);

        current_frame->erase (firstPkt);
        this->pktsHistory[m_netGlobalId] = pkt_info;
        this->m_send_wnd.push_back (m_netGlobalId);
        this->m_goodput_wnd.push_back (m_netGlobalId);

        while(!this->m_goodput_wnd.empty()){
            uint16_t id_out_of_date = this->m_goodput_wnd.front();
            //NS_ASSERT_MSG(this->pktsHistory.find(id_out_of_date)!=this->pktsHistory.end(),"History should record every packet sent.");
            Ptr<SentPacketInfo> info_out_of_date = this->pktsHistory[id_out_of_date];
            if((uint64_t)(info_out_of_date->pkt_send_time.GetMicroSeconds()) < NowUs - this->m_goodput_wnd_size){
                this->total_pkts_inwnd -= info_out_of_date->pkt_size;
                if(info_out_of_date->is_goodput){
                    this->goodput_pkts_inwnd -= info_out_of_date->pkt_size;
                }
                this->m_goodput_wnd.pop_front();
            }
            else{
                break;
            }
        }

        while(this->m_send_wnd.size()>this->m_send_wnd_size){
            uint16_t id_out_of_date = this->m_send_wnd.front();
            NS_ASSERT_MSG(this->pktsHistory.find(id_out_of_date)!=this->pktsHistory.end(),"History should record every packet sent.");
            this->pktsHistory.erase(id_out_of_date);
            this->m_send_wnd.pop_front();
        }

        m_netGlobalId = (m_netGlobalId + 1) % 65536;

        if(current_frame->empty()){
            this->m_queue.erase(this->m_queue.begin());
            this->Calculate_pacing_rate();
        }
        if(m_pacing){
            this->m_pacingTimer.Schedule();
        }
        else {
            SendPacket ();
        }

    }
};


uint32_t PacketSender::num_frame_in_queue()
{
    return this->m_queue.size();
}

TypeId PacketFrame::GetTypeId()
{
    static TypeId tid = TypeId ("ns3::PacketFrame")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

PacketFrame::PacketFrame(std::vector<Ptr<VideoPacket>> packets, bool retransmission)
{
    this->packets_in_Frame.assign(packets.begin(), packets.end());
    this->retransmission = retransmission;
};

PacketFrame::~PacketFrame() {};

uint32_t PacketFrame::Frame_size_in_byte()
{
    return this->packets_in_Frame.size() * this->per_packet_size;
};

uint32_t PacketFrame::Frame_size_in_packet()
{
    return this->packets_in_Frame.size();
};

SentPacketInfo::SentPacketInfo(uint16_t id, uint16_t batch_id, Time sendtime, PacketType type, bool isgoodput, uint16_t size){
    this->pkt_id = id;
    this->batch_id = batch_id;
    this->pkt_send_time = sendtime;
    this->pkt_ack_time = MicroSeconds(0);
    this->pkt_type = type;
    this->is_goodput = isgoodput;
    this->pkt_size = size;
};

SentPacketInfo::~SentPacketInfo() {};

}; // namespace ns3
