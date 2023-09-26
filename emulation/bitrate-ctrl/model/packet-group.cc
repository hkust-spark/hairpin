#include "packet-group.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("PacketBatch");

TypeId PacketBatch::GetTypeId() {
    static TypeId tid = TypeId ("ns3::PacketBatch")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

PacketBatch::PacketBatch(Ptr<VideoPacket> pkt, Ptr<PacketGroup> packet_group) {
    this->packet_group = packet_group;
    // Batch info
    this->batch_id = pkt->GetBatchId();
    this->data_num = pkt->GetBatchDataNum();
    this->fec_num = pkt->GetBatchFECNum();
    NS_ASSERT_MSG(this->data_num>0, "Must have at least 1 packet in PacketBatch");
    // Add packet
    this->AddPacket(pkt);
};

PacketBatch::~PacketBatch() {

};

void PacketBatch::AddPacket(Ptr<VideoPacket> pkt) {
    uint16_t pkt_id = pkt->GetPktIdBatch();
    if(this->decoded_pkts.find(pkt_id) != this->decoded_pkts.end())
        return;

    auto packet_type = pkt->GetPacketType();
    if(packet_type == PacketType::DATA_PKT) {
        // data packet, can be passed to decoder directly
        Ptr<DataPacket> data_pkt = DynamicCast<DataPacket, VideoPacket> (pkt);
        this->data_pkts[pkt_id] = data_pkt;
        this->undecoded_pkts.push_back(data_pkt);
    } else if(packet_type == PacketType::DUP_FEC_PKT) {
        // actually the same as data packet
        Ptr<DupFECPacket> dup_fec_pkt = DynamicCast<DupFECPacket, VideoPacket> (pkt);
        this->data_pkts[pkt_id] = dup_fec_pkt;
        this->undecoded_pkts.push_back(dup_fec_pkt);
    } else if(packet_type == PacketType::FEC_PKT)
        this->fec_pkts[pkt_id] = DynamicCast<FECPacket, VideoPacket> (pkt);
};

bool PacketBatch::CheckComplete() {
    // DEBUG("[PacketBatch] batch id: " << this->batch_id << ", batch_data_num: " << this->data_num << ", current data pkt num:" << this->data_pkts.size() << ", current fec pkt num:" << this->fec_pkts.size());
    // FEC calculation here
    // DEBUG("data size: " << this->data_pkts.size() << ", FEC size: " << this->fec_pkts.size() << ", DATA num: " << this->data_num <<"\n";

    if(this->data_pkts.size() + this->fec_pkts.size() >= this->data_num) {
        if(this->data_pkts.size() == this->data_num){
            // all data packets arrived
            return true;
        } else {
            // recover data from FEC
            // Assume every FEC packet stores data of all other data packets
            NS_ASSERT(this->fec_pkts.size() > 0);
            Time now = Simulator::Now();
            auto data_pkt_digests = this->fec_pkts.begin()->second->GetDataPacketDigests();
            for(auto data_pkt_digest : data_pkt_digests) {
                // std::cout << "[PacketBatch] in FEC packets: pkt_id_in_batch: " << 
                //     data_pkt_digest->pkt_id_in_batch << "pkt_id_in_group: " << 
                //     data_pkt_digest->pkt_id_in_group <<'\n';
                if(this->data_pkts.find(data_pkt_digest->pkt_id_in_batch) == this->data_pkts.end()) {
                    // data packet not exists in this->data_pkts
                    Ptr<DataPacket> data_pkt = Create<DataPacket> (
                        data_pkt_digest,
                        this->packet_group->GetGroupId(), this->packet_group->GetDataNum(), this->packet_group->GetFECNum(),
                        this->batch_id, this->data_num, this->fec_num
                    );
                    data_pkt->SetRcvTime(now);
                    data_pkt->SetEncodeTime(this->fec_pkts.begin()->second->GetEncodeTime());
                    // std::cout << "[PacketBatch] group id: " << this->packet_group->GetGroupId() << ", batch id: " << this->batch_id <<
                    //     ", pkt id: " << data_pkt->GetPktIdGroup());
                    this->undecoded_pkts.push_back(data_pkt);
                }
            }
            this->data_pkts.clear();
            this->fec_pkts.clear();
            return true;
        }
    }
    return false;
};

std::vector<Ptr<DataPacket>> PacketBatch::GetUndecodedPackets() {
    auto data_pkt_list = this->undecoded_pkts;
    // clear this->undecoded_pkts
    this->undecoded_pkts.clear();
    // mark data packets as decoded
    for(auto pkt : data_pkt_list)
        this->decoded_pkts.insert(pkt->GetPktIdBatch());
    // return those undecoded data packets
    return data_pkt_list;
};

TypeId PacketGroup::GetTypeId() {
    static TypeId tid = TypeId ("ns3::PacketGroup")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

PacketGroup::PacketGroup(uint32_t group_id, uint16_t data_num, Time encode_time) {
    // Group related
    this->group_id = group_id;
    this->data_num = data_num;
    this->encode_time = encode_time;
    // statistics
    this->max_tx_count = 0;  
    this->first_tx_data_count = 0;
    this->first_tx_fec_count = 0;
    this->rtx_data_count = 0;
    this->rtx_fec_count = 0;
    this->next_rtx_count = 1;
};

PacketGroup::PacketGroup(Ptr<VideoPacket> pkt, Time group_delay) {
    // Group related
    this->group_id = pkt->GetGroupId();
    this->data_num = pkt->GetGroupDataNum();
    this->encode_time = pkt->GetEncodeTime();
    // statistics
    this->first_tx_data_count = 0;
    this->first_tx_fec_count = 0;
    this->rtx_data_count = 0;
    this->rtx_fec_count = 0;
    this->next_rtx_count = 1;
    // Add packet
    this->AddPacket(pkt, group_delay);
};

PacketGroup::~PacketGroup() {

};

void PacketGroup::InitGroup(Ptr<VideoPacket> pkt) {
    this->fec_num = pkt->GetGroupFECNum();
    this->first_rcv_time = pkt->GetRcvTime();
    this->last_rcv_time_tx_0 = pkt->GetRcvTime();
    this->pkt_cnt_tx_0 = 0;
    this->max_tx_count = pkt->GetTXCount();  
    this->last_pkt_id_tx_0 = pkt->GetPktIdGroup();
};

uint32_t PacketGroup::GetGroupId() { return this->group_id; };
uint16_t PacketGroup::GetDataNum() { return this->data_num; };
uint16_t PacketGroup::GetFECNum() { return this->fec_num; };
Time PacketGroup::GetEncodeTime() { return this->encode_time; };
Time PacketGroup::GetLastRcvTimeTx0() { return this->last_rcv_time_tx_0; };
Time PacketGroup::GetAvgPktInterval() {
    if(last_pkt_id_tx_0 == 0 || pkt_cnt_tx_0 <= 1) return MicroSeconds(0);
    return (last_rcv_time_tx_0 - first_rcv_time) / (this->pkt_cnt_tx_0 - 1);
};

uint8_t PacketGroup::GetMaxTxCount() {
    return this->max_tx_count;
};

uint16_t PacketGroup::GetFirstTxDataCount() { return this->first_tx_data_count; };

uint16_t PacketGroup::GetFirstTxFECCount() { return this->first_tx_fec_count; };

uint16_t PacketGroup::GetRtxDataCount() { return this->rtx_data_count; };

uint16_t PacketGroup::GetRtxFECCount() { return this->rtx_fec_count; };

std::unordered_map<uint8_t, uint32_t> PacketGroup::GetPktArriveDistribution() { return this->pkt_arrive_dis; };


void PacketGroup::AddPacket(Ptr<VideoPacket> pkt, Time group_delay) {

    if(this->decoded_pkts.size() + this->undecoded_pkts.size() == 0)
        this->InitGroup(pkt);

    // statistics
    auto packet_type = pkt->GetPacketType();
    if(pkt->GetTXCount() == 0) {
        if(packet_type == PacketType::DATA_PKT) this->first_tx_data_count ++;
        else if(packet_type == PacketType::FEC_PKT
            || packet_type == PacketType::DUP_FEC_PKT) this->first_tx_fec_count ++;
    } else {
        if(packet_type == PacketType::DATA_PKT) this->rtx_data_count ++;
        else if(packet_type == PacketType::FEC_PKT
            || packet_type == PacketType::DUP_FEC_PKT) this->rtx_fec_count ++;
    }

    // if(packet_type == DUP_FEC_PKT) {
    //     DEBUG("[PacketGroup] DUP_FEC rcvd group id: " << this->group_id << ", pkt_id_in_group: " << pkt->GetPktIdGroup());
    // }

    if(pkt->GetTXCount() > this->max_tx_count) this->max_tx_count = pkt->GetTXCount();
    if(pkt->GetEncodeTime() < this->encode_time) this->encode_time = pkt->GetEncodeTime();
    if(pkt->GetRcvTime() > this->last_rcv_time_tx_0 && pkt->GetTXCount() == 0) {
        this->last_rcv_time_tx_0 = pkt->GetRcvTime();
        this->pkt_cnt_tx_0 ++;
    }
    if(pkt->GetPktIdGroup() > this->last_pkt_id_tx_0 && pkt->GetPktIdGroup() < data_num + fec_num && pkt->GetTXCount() == 0) this->last_pkt_id_tx_0 = pkt->GetPktIdGroup();

    uint32_t batch_id = pkt->GetBatchId();
    // if it's a useless packet for decoding, return
    if(this->complete_batches.find(batch_id) != this->complete_batches.end()) return;
    if(this->decoded_pkts.find(pkt->GetPktIdGroup()) != this->decoded_pkts.end()) return;

    // insert it into PacketBatch
    if(this->incomplete_batches.find(batch_id) == this->incomplete_batches.end())
        this->incomplete_batches[batch_id] = Create<PacketBatch> (pkt, this);
    else
        this->incomplete_batches[batch_id]->AddPacket(pkt);

    // check if the batch is complete
    bool batch_complete = this->incomplete_batches[batch_id]->CheckComplete();
    // Get undecoded data packets and store it
    auto batch_undecoded_pkts = this->incomplete_batches[batch_id]->GetUndecodedPackets();
    this->undecoded_pkts.insert(
        this->undecoded_pkts.end(), batch_undecoded_pkts.begin(), batch_undecoded_pkts.end()
    );
    // If complete, move the batch to complete_batches
    if(batch_complete) {
        this->complete_batches[batch_id] = this->incomplete_batches[batch_id];
        this->incomplete_batches.erase(batch_id);
    }
};


std::vector<Ptr<DataPacket>> PacketGroup::GetUndecodedPackets() {
    auto data_pkt_list = this->undecoded_pkts;
    // clear this->undecoded_pkts
    this->undecoded_pkts.clear();
    // mark data packets as decoded
    for(auto pkt : data_pkt_list) {
        this->decoded_pkts.insert(pkt->GetPktIdGroup());
        if(this->pkt_arrive_dis.find(pkt->GetTXCount()) == this->pkt_arrive_dis.end())
            this->pkt_arrive_dis[pkt->GetTXCount()] = 0;
        this->pkt_arrive_dis[pkt->GetTXCount()] ++;
    }
    // return those undecoded data packets
    return data_pkt_list;
};

bool PacketGroup::CheckComplete() {
    if(this->decoded_pkts.size() == this->data_num)
        return true;
    return false;
};

}; // namespace ns3