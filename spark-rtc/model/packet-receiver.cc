#include "packet-receiver.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("PacketReceiver");

TypeId PacketReceiver::GetTypeId() {
    static TypeId tid = TypeId ("ns3::PacketReceiver")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

PacketReceiver::PacketReceiver (GameClient * game_client, void (GameClient::*RcvPacketFunc)(Ptr<VideoPacket>), Ptr<Socket> socket, uint32_t wndsize)
: game_client{game_client}
, m_socket {socket}
, RcvPacketFunc {RcvPacketFunc}
, m_record {}
, m_feedback_interval {MilliSeconds(16)}
, m_feedbackTimer {Timer::CANCEL_ON_DESTROY}
, m_last_feedback {65535}
, wnd_size {1000}
, time_wnd_size{wndsize}
, m_credible {1000}
, bytes_per_packet {1500}
, last_id {0}
, pkts_in_wnd {0}
, bytes_in_wnd {0}
, time_in_wnd {0}
, losses_in_wnd {0}
, loss_rate {0.}
, throughput_kbps {30000.}
, one_way_dispersion {MicroSeconds(180)}
, m_loss_seq {1000}
, m_recv_sample {}
{
    this->m_socket->SetRecvCallback(MakeCallback(&PacketReceiver::OnSocketRecv_receiver,this));
    this->m_feedbackTimer.SetFunction(&PacketReceiver::Feedback_NetState,this);
    this->m_feedbackTimer.SetDelay(m_feedback_interval);
};

PacketReceiver::~PacketReceiver () {};

void PacketReceiver::OnSocketRecv_receiver(Ptr<Socket> socket)
{
    Time time_now = Simulator::Now();
    Ptr<Packet> pkt = socket->Recv();
    Ptr<NetworkPacket> net_pkt = NetworkPacket::ToInstance(pkt);
    net_pkt->SetRcvTime(time_now);
    PacketType pkt_type = net_pkt->GetPacketType();

    if(pkt_type != DATA_PKT && pkt_type != FEC_PKT && pkt_type != DUP_FEC_PKT)
        // not data or FEC packets
        return;

    Ptr<VideoPacket> video_pkt = DynamicCast<VideoPacket, NetworkPacket> (net_pkt);
    ((this->game_client)->*RcvPacketFunc)(video_pkt);

    /* update network statistics */
    uint32_t id = video_pkt->GetGlobalId();
    uint32_t RxTime = time_now.GetMicroSeconds();
    Ptr<RcvTime> rt = Create<RcvTime>(id, RxTime, pkt->GetSize());

    if(this->m_record.empty()) {
        this->m_record.push_back(rt);
        this->last_id = id;
        if(this->m_feedbackTimer.IsExpired()){
            this->m_feedbackTimer.Schedule();
        }
    }
    else
    {
        // slide on the window
        std::deque<Ptr<RcvTime>>::iterator iter;
        for(iter = this->m_record.begin();iter<this->m_record.end();){
            if((*iter)->rt_us < RxTime - this->time_wnd_size){
                this->bytes_in_wnd -= (*iter)->pkt_size;
                iter = this->m_record.erase(iter);
            } else iter++;
        }
        // while(this->m_record.front()->rt_us < RxTime - this->time_wnd_size) {
        //     this->bytes_in_wnd -= this->m_record.front()->pkt_size;
        //     this->m_record.pop_front();
        //     if(this->m_record.empty()) break;
        // }

        // insert new packet
        if(this->m_record.empty()){
            this->m_record.push_back(rt);
        }
        else {
            if(this->lessThan_simple(id, this->m_record.front()->pkt_id)){
                this->m_record.push_front(rt);
            }
            else{
                std::deque<Ptr<RcvTime>>::iterator it = this->m_record.end();
                while(it > this->m_record.begin()){
                    if(this->lessThan_simple((*(it-1))->pkt_id, id)){
                        this->m_record.insert(it, rt);
                        break;
                    }
                    it--;
                }
            }

        }


        // if(lessThan_simple(id, this->m_last_feedback)){
        //     this->m_recv_sample.push_back(rt);
        // }
    }
    // DEBUG("[Rcver] throughput (kbps):" << (float)this->bytes_in_wnd / (float)this->time_wnd_size * 8. * 1000. << ", time wnd size:" <<this->time_wnd_size<<" ms\n";
    this->bytes_in_wnd += pkt->GetSize();
    // std::cout << "append pkt size:" << pkt->GetSize() << ", bytes in wnd:" << this->bytes_in_wnd << ", pkts in wnd:" << this->m_record.size() - 1);

};

void PacketReceiver::Feedback_NetState ()
{
    this->pkts_in_wnd = 0;
    if(this->m_record.size() > 1) {
        Ptr<RcvTime> record_end = m_record.back();
        Ptr<RcvTime> record_front = m_record.front();
        this->pkts_in_wnd = (record_end->pkt_id - record_front->pkt_id + 65537) % 65536;
        //NS_ASSERT_MSG(this->pkts_in_wnd <= this->wnd_size + 1,"Packets in window "<<this->pkts_in_wnd<<" is larger than window size limit "<<(this->wnd_size + 1));
        //this->time_in_wnd = record_end->rt_us - record_front->rt_us;
    }
    //this->bytes_in_wnd = this->m_record.size() * this->bytes_per_packet;
    this->losses_in_wnd = 0;

    // if(this->time_in_wnd > 0) {
    //     this->throughput_kbps = (float)this->bytes_in_wnd / (float)this->time_in_wnd * 8. * 1000.;
    // }
    this->throughput_kbps = (float)this->bytes_in_wnd / (float)this->time_wnd_size * 8. * 1000.;
    // std::cout << "[Rcver] throughput (kbps):" << this->throughput_kbps<<", bytes in wnd:" <<this->bytes_in_wnd << ", pkts in wnd:" << this->pkts_in_wnd << ", time wnd size:" <<this->time_wnd_size<<" ms\n";

    // if(throughput_kbps > 100000){
    //     std::cout << "throughput(kbps):" << this->throughput_kbps<<", bytes in wnd:" <<this->bytes_in_wnd
    //               << ", pkts in wnd:" << this->pkts_in_wnd << ", time wnd size:" <<this->time_wnd_size<<"\n";
    // }

    //Gather m_loss_seq & m_recvtime_sample from m_record
    this->m_loss_seq.clear();
    this->m_recv_sample.clear();
    uint32_t prev_id = 0;

    auto it = this->m_record.begin();
    int consecutive_recv = 0;
    uint32_t id;
    if(!this->m_record.empty()){
        prev_id = (*it)->pkt_id - 1;
        bool first=true;
        while (it<this->m_record.end())
        {
            id = (*it)->pkt_id;

            if(!first){
                NS_ASSERT_MSG(this->lessThan_simple(prev_id,id),"history should be ordered. Previous ID="<<prev_id<<" while ID="<<id<<" with last_id="<<this->last_id);
            }
            else{
                first=false;
            }

            if ((id - prev_id + 65536)%65536 == 1) {
                consecutive_recv += 1;
                if(it == this->m_record.end()-1) {this->m_loss_seq.push_back(consecutive_recv);}
            }
            else {
                this->m_loss_seq.push_back(consecutive_recv);
                this->m_loss_seq.push_back(1 - (id - prev_id + 65536)%65536);
                consecutive_recv = 1;
                this->losses_in_wnd += id - prev_id - 1;
            }
            if(this->lessThan_simple(this->m_last_feedback, id)){
                this->m_recv_sample.push_back(*it);
                this->m_last_feedback = id;
            }

            it++;
            prev_id = id;
        }
    }

    if(this->pkts_in_wnd >= 1){
        this->loss_rate = (float)this->losses_in_wnd / (float)this->pkts_in_wnd;
    }

    NS_LOG_FUNCTION("LossRate" << this->loss_rate << "Throughput" << this->throughput_kbps << "FEC group delay" << this->one_way_dispersion.GetMicroSeconds());

    /* construct NetState*/
    Ptr<NetStates> netstate = Create<NetStates>(this->loss_rate, (uint32_t)this->throughput_kbps, (uint16_t)this->one_way_dispersion.GetMicroSeconds());
    netstate->loss_seq = m_loss_seq;
    netstate->recvtime_hist = m_recv_sample;

    Ptr<NetStatePacket> nstpacket = Create<NetStatePacket>();
    nstpacket->SetNetStates(netstate);
    this->m_socket->Send(nstpacket->ToNetPacket());
    // this->m_recv_sample.clear();
    this->m_feedbackTimer.Schedule();
};


void PacketReceiver::SendPacket(Ptr<NetworkPacket> pkt) {
    this->m_socket->Send(pkt->ToNetPacket());
}

void PacketReceiver::Set_FECgroup_delay(Time& fec_group_delay_)
{
    this->one_way_dispersion = fec_group_delay_;
};
Time PacketReceiver::Get_FECgroup_delay()
{
    return this->one_way_dispersion;
};

void PacketReceiver::StopRunning() {
    if(this->m_feedbackTimer.IsRunning()){
        this->m_feedbackTimer.Cancel();
    }
};

bool PacketReceiver::lessThan(uint16_t id1, uint16_t id2) {
    //NS_ASSERT_MSG(((this->last_id - id1 + 65536) % 65536) <= this->wnd_size &&
    //              ((this->last_id - id2 + 65536) % 65536) <= this->wnd_size,
    //              "ID1 and ID2 must be in-window.");
    if(this->last_id >= this->wnd_size && id1 < id2)
        return true;

    if(this->last_id<this->wnd_size) {
        if(id2<=this->last_id && id1>=65536-this->wnd_size+this->last_id)
            return true;
        if(id1<id2) {
            if((id1 >= 65536 - this->wnd_size + this->last_id) &&
               (id2 >= 65536 - this->wnd_size + this->last_id))
                return true;
            if((id1 <= this->last_id) &&
               (id2 <= this->last_id))
                return true;
        }
    }
    return false;
};

bool PacketReceiver::lessThan_simple(uint16_t id1, uint16_t id2){
    uint16_t noWarpSubtract = id2 - id1;
    uint16_t wrapSubtract = id1 - id2;
    return noWarpSubtract < wrapSubtract;
}

}; // namespace ns3