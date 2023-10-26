#include "network-packet.h"

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("NetworkPacket");

NS_OBJECT_ENSURE_REGISTERED (NetworkPacket);

/* class NetworkPacket */
TypeId NetworkPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetworkPacket")
        .SetParent<Object> ()
        .SetGroupName ("spark-rtc")
        .AddAttribute ("MaxPacketSize", "The size of the packet",
                       UintegerValue (1460),
                       MakeUintegerAccessor (&NetworkPacket::MAX_PACKET_SIZE),
                       MakeUintegerChecker<uint16_t> ())
    ;
    return tid;
};

NetworkPacket::NetworkPacket(PacketType packet_type) {
    this->network_header.packet_type = packet_type;
    this->network_payload.payload_size = 0;
    this->send_time = MicroSeconds(0);
    this->rcv_time = MicroSeconds(0);
};

NetworkPacket::~NetworkPacket() {};

void NetworkPacket::SetSendTime(Time time) { this->send_time = time; };
void NetworkPacket::SetRcvTime(Time time) { this->rcv_time = time; };

Time NetworkPacket::GetSendTime() { return this->send_time; };
bool NetworkPacket::IsPacketSent() { return this->send_time != MicroSeconds(-1); };
Time NetworkPacket::GetRcvTime() { return this->rcv_time; };


PacketType NetworkPacket::GetPacketType() { return this->network_header.packet_type; };
void NetworkPacket::SetPacketType(PacketType packet_type) { this->network_header.packet_type = packet_type ; };

uint16_t NetworkPacket::GetMaxPayloadSize() {
    Ptr<NetworkPacket> netPkt = CreateObject<NetworkPacket> (PacketType::DATA_PKT);
    UintegerValue maxPacketSize;
    netPkt->GetAttribute ("MaxPacketSize", maxPacketSize);
    return maxPacketSize.Get () - 4; 
};

uint32_t NetworkPacket::GetPayloadSize() { return this->network_payload.payload_size; };

uint8_t * NetworkPacket::GetPayloadPtr() { return this->network_payload.payload_buffer; };

void NetworkPacket::SetPayload(uint8_t * buffer, uint32_t size) {
    NS_ASSERT(size <= NetworkPacket::GetMaxPayloadSize());
    this->network_payload.payload_buffer = buffer;
    this->network_payload.payload_size = size;
}

// static
Ptr<NetworkPacket> NetworkPacket::ToInstance(Ptr<Packet> packet) {
    NetworkPacketHeader network_header = NetworkPacketHeader();
    packet->RemoveHeader(network_header);
    switch (network_header.packet_type)
    {
    case PacketType::DATA_PKT:
        return Create<DataPacket> (packet);
    case PacketType::DUP_FEC_PKT:
        return Create<DupFECPacket> (packet);
    case PacketType::FEC_PKT:
        return Create<FECPacket> (packet);
    case PacketType::ACK_PKT:
        return Create<AckPacket> (packet);
    case PacketType::FRAME_ACK_PKT:
        return Create<FrameAckPacket> (packet);
    case PacketType::NETSTATE_PKT:
        return Create<NetStatePacket> (packet);
    default:
        return nullptr; //casually defined
    }
}

Ptr<Packet> NetworkPacket::ToNetPacket () {
    NS_FATAL_ERROR ("Virtual function called");
};

/* class VideoPacket */
TypeId VideoPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::VideoPacket")
        .SetParent<NetworkPacket> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

VideoPacket::VideoPacket(PacketType packet_type) : NetworkPacket(packet_type) {
    this->video_header.tx_count = 0;
    this->batch_info_set_flag = false;
    this->group_info_set_flag = false;
    this->video_header.encode_time = MicroSeconds(0);
    this->enqueue_time = MicroSeconds(0);
};

VideoPacket::~VideoPacket() {};

void VideoPacket::SetEncodeTime(Time time) { this->video_header.encode_time = time; };
void VideoPacket::SetEnqueueTime(Time time) { this->enqueue_time = time; };
Time VideoPacket::GetEncodeTime() { return this->video_header.encode_time; };
Time VideoPacket::GetEnqueueTime() { return this->enqueue_time; };

void VideoPacket::SetFECGroup(uint32_t group_id, u_int16_t group_data_num, uint16_t group_fec_num, uint16_t pkt_id_in_group) {
    this->video_header.group_id = group_id;
    this->video_header.group_data_num = group_data_num;
    this->video_header.group_fec_num = group_fec_num;
    this->video_header.pkt_id_in_group = pkt_id_in_group;
    this->group_info_set_flag = true;
};

void VideoPacket::SetFECBatch(uint32_t batch_id, u_int16_t batch_data_num, uint16_t batch_fec_num, uint16_t pkt_id_in_batch) {
    this->video_header.batch_id = batch_id;
    this->video_header.batch_data_num = batch_data_num;
    this->video_header.batch_fec_num = batch_fec_num;
    this->video_header.pkt_id_in_batch = pkt_id_in_batch;
    this->batch_info_set_flag = true;
};

void VideoPacket::ClearFECBatch() {
    this->batch_info_set_flag = false;
};

void VideoPacket::SetTXCount(uint8_t tx_count) { this->video_header.tx_count = tx_count; };
void VideoPacket::IncreTXCount() { this->video_header.tx_count ++; };

void VideoPacket::SetGlobalId(uint16_t id) {this->video_header.global_id = id;};
uint16_t VideoPacket::GetGlobalId() { return this->video_header.global_id; };
bool VideoPacket::GetGroupInfoSetFlag() { return this->group_info_set_flag; };
uint32_t VideoPacket::GetGroupId() { return this->video_header.group_id; };
uint16_t VideoPacket::GetGroupDataNum() { return this->video_header.group_data_num; };
uint16_t VideoPacket::GetGroupFECNum() { return this->video_header.group_fec_num; };
uint16_t VideoPacket::GetPktIdGroup() { return this->video_header.pkt_id_in_group; };
bool VideoPacket::GetBatchInfoSetFlag() { return this->batch_info_set_flag; };
uint32_t VideoPacket::GetBatchId() { return this->video_header.batch_id; };
uint16_t VideoPacket::GetBatchDataNum() { return this->video_header.batch_data_num; };
uint16_t VideoPacket::GetBatchFECNum() { return this->video_header.batch_fec_num; };
uint16_t VideoPacket::GetPktIdBatch() { return this->video_header.pkt_id_in_batch; };
uint8_t VideoPacket::GetTXCount() { return this->video_header.tx_count; };

uint16_t VideoPacket::GetMaxPayloadSize() { return NetworkPacket::GetMaxPayloadSize () - 38; };

Ptr<Packet> VideoPacket::ToNetPacket () {
    NS_FATAL_ERROR ("Virtual function called");
}

DataPktDigest::DataPktDigest() {};

DataPktDigest::DataPktDigest(Ptr<DataPacket> pkt) {
    this->pkt_id_in_batch = pkt->GetPktIdBatch();
    this->pkt_id_in_group = pkt->GetPktIdGroup();
    this->frame_id = pkt->GetFrameId();
    this->frame_pkt_num = pkt->GetFramePktNum();
    this->pkt_id_in_frame = pkt->GetPktIdFrame();
};

/* class DataPacket */
TypeId DataPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DataPacket")
        .SetParent<VideoPacket> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

DataPacket::DataPacket(uint32_t frame_id, uint16_t frame_pkt_num, uint16_t pkt_id_in_frame)
     : VideoPacket(PacketType::DATA_PKT) {
    this->data_header.last_pkt_mark = false;
    this->SetFrameInfo(frame_id, frame_pkt_num, pkt_id_in_frame);
};

DataPacket::DataPacket(Ptr<Packet> packet) : VideoPacket(PacketType::DATA_PKT) {
    // NetworkPacketHeader has been removed in NetworkPacket::ToInstance
    packet->RemoveHeader(this->video_header);
    packet->RemoveHeader(this->data_header);
    //packet->RemoveTrailer(this->network_payload);
};

DataPacket::~DataPacket() {};

Ptr<Packet> DataPacket::ToNetPacket() {
    uint32_t packet_size = this->GetPayloadSize();
    Ptr<Packet> packet = Create<Packet> (packet_size);
    packet->AddHeader(this->data_header);
    packet->AddHeader(this->video_header);
    packet->AddHeader(this->network_header);
    //if(this->GetPayloadSize() != 0)
    //    packet->AddTrailer(this->network_payload);
    return packet;
};

DataPacket::DataPacket(Ptr<DataPktDigest> data_pkt_digest,
    uint32_t group_id, uint16_t group_data_num, uint16_t group_fec_num,
    uint32_t batch_id, uint16_t batch_data_num, uint16_t batch_fec_num) : VideoPacket(PacketType::DATA_PKT) {
    this->data_header.frame_id = data_pkt_digest->frame_id;
    this->data_header.frame_pkt_num = data_pkt_digest->frame_pkt_num;
    this->data_header.pkt_id_in_frame = data_pkt_digest->pkt_id_in_frame;
    this->SetFECBatch(batch_id, batch_data_num, batch_fec_num, data_pkt_digest->pkt_id_in_batch);
    this->SetFECGroup(group_id, group_data_num, group_fec_num, data_pkt_digest->pkt_id_in_group);
};

uint16_t DataPacket::GetMaxPayloadSize() { return VideoPacket::GetMaxPayloadSize() - 8; };

void DataPacket::SetFrameInfo(uint32_t frame_id, uint16_t frame_pkt_num, uint16_t pkt_id_in_frame) {
    this->data_header.frame_id = frame_id;
    this->data_header.frame_pkt_num = frame_pkt_num;
    this->data_header.pkt_id_in_frame = pkt_id_in_frame;
};

void DataPacket::SetLastPkt(bool last_pkt_mark) { this->data_header.last_pkt_mark = last_pkt_mark; };

uint32_t DataPacket::GetFrameId() { return this->data_header.frame_id; };
uint16_t DataPacket::GetFramePktNum() { return this->data_header.frame_pkt_num; };
uint16_t DataPacket::GetPktIdFrame() { return this->data_header.pkt_id_in_frame; };
bool DataPacket::GetLastPktMark() { return this->data_header.last_pkt_mark; };
void DataPacket::SetDataGlobalId (uint16_t id) { m_dataGlobalId = id; };
uint16_t DataPacket::GetDataGlobalId () { return m_dataGlobalId; };


/* class DupFECPacket */
TypeId DupFECPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DupFECPacket")
        .SetParent<DataPacket> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

DupFECPacket::DupFECPacket(Ptr<DataPacket> data_pkt) : DataPacket(*data_pkt) {
    this->SetPacketType(DUP_FEC_PKT);
};
DupFECPacket::DupFECPacket(Ptr<Packet> pkt) : DataPacket(pkt) {
    this->SetPacketType(DUP_FEC_PKT);
};

DupFECPacket::~DupFECPacket() {};

DupFECPacket::DupFECPacket(uint32_t frame_id, uint16_t frame_pkt_num, uint16_t pkt_id_in_frame)
     : DataPacket(frame_id, frame_pkt_num, pkt_id_in_frame) {
    this->SetPacketType(PacketType::DUP_FEC_PKT);
};


/* class FECPacket */
TypeId FECPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FECPacket")
        .SetParent<VideoPacket> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

FECPacket::FECPacket(uint8_t tx_count, std::vector<Ptr<DataPacket>> data_pkts) : VideoPacket(PacketType::FEC_PKT) {
    this->SetTXCount(tx_count);
    for(auto data_pkt : data_pkts) {
        this->fec_header.data_pkts.push_back(Create<DataPktDigest> (data_pkt));
    }
};

FECPacket::FECPacket(Ptr<Packet> packet) : VideoPacket(PacketType::FEC_PKT) {
    // NetworkPacketHeader has been removed in NetworkPacket::ToInstance
    packet->RemoveHeader(this->video_header);
    packet->RemoveHeader(this->fec_header);
    //packet->RemoveTrailer(this->network_payload);
};

FECPacket::~FECPacket() {};

Ptr<Packet> FECPacket::ToNetPacket() {
    Ptr<Packet> packet = Create<Packet> ( - this->GetHeaderLength() + VideoPacket::GetMaxPayloadSize());
    packet->AddHeader(this->fec_header);
    packet->AddHeader(this->video_header);
    packet->AddHeader(this->network_header);
    //if(this->GetPayloadSize() != 0)
    //    packet->AddTrailer(this->network_payload);
    return packet;
};

uint32_t FECPacket::GetHeaderLength() {
    return 2 /* size */ + this->fec_header.data_pkts.size() * (2 + 2 + 4 + 2 + 2);
};

std::vector<Ptr<DataPktDigest>> FECPacket::GetDataPacketDigests() { return this->fec_header.data_pkts; };

void FECPacket::SetDataPackets(std::vector<Ptr<DataPacket>> data_pkts) {
    for(auto data_pkt : data_pkts) {
        this->fec_header.data_pkts.push_back(Create<DataPktDigest> (data_pkt));
    }
};


/* class ControlPacket */
TypeId ControlPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::ControlPacket")
        .SetParent<NetworkPacket> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

ControlPacket::ControlPacket(PacketType packet_type) : NetworkPacket(packet_type) {};

ControlPacket::~ControlPacket() {};

Ptr<Packet> ControlPacket::ToNetPacket () {
    NS_FATAL_ERROR ("Virtual function called");
}

/* class AckPacket */
TypeId AckPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::AckPacket")
        .SetParent<ControlPacket> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<AckPacket> ()
    ;
    return tid;
};

AckPacket::AckPacket() : ControlPacket(PacketType::ACK_PKT) {

};

AckPacket::AckPacket(Ptr<Packet> packet) : ControlPacket(PacketType::ACK_PKT) {
    // NetworkPacketHeader has been removed in NetworkPacket::ToInstance
    packet->RemoveHeader(this->ack_header);
    //packet->RemoveTrailer(this->network_payload);
};

AckPacket::AckPacket(std::vector<Ptr<GroupPacketInfo>> pkt_infos, uint16_t last_pkt_id) : ControlPacket(PacketType::ACK_PKT) {
    for(auto pkt_info : pkt_infos) {
        Ptr<GroupPacketInfo> pkt = Create<GroupPacketInfo> (*pkt_info);
        this->ack_header.pkt_infos.push_back(pkt);
    }
    this->ack_header.last_pkt_id = last_pkt_id;
};

AckPacket::~AckPacket() {};

Ptr<Packet> AckPacket::ToNetPacket() {
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader(this->ack_header);
    packet->AddHeader(this->network_header);
    //if(this->GetPayloadSize() != 0)
    //    packet->AddTrailer(this->network_payload);
    return packet;
};

std::vector<Ptr<GroupPacketInfo>> AckPacket::GetAckedPktInfos() {
    return this->ack_header.pkt_infos;
};

uint16_t AckPacket::GetLastPktId() {
    return this->ack_header.last_pkt_id;
};


/* class FrameAckPacket */
TypeId FrameAckPacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FrameAckPacket")
        .SetParent<ControlPacket> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<FrameAckPacket> ()
    ;
    return tid;
};

FrameAckPacket::FrameAckPacket() : ControlPacket(PacketType::FRAME_ACK_PKT) {

};

FrameAckPacket::FrameAckPacket(Ptr<Packet> packet) : ControlPacket(PacketType::FRAME_ACK_PKT) {
    // NetworkPacketHeader has been removed in NetworkPacket::ToInstance
    packet->RemoveHeader(this->ack_header);
    //packet->RemoveTrailer(this->network_payload);
};

FrameAckPacket::FrameAckPacket(uint32_t frame_id, Time frame_encode_time) : ControlPacket(PacketType::FRAME_ACK_PKT) {
    this->ack_header.frame_id = frame_id;
    this->ack_header.frame_encode_time = frame_encode_time;
};

FrameAckPacket::~FrameAckPacket() {};

Ptr<Packet> FrameAckPacket::ToNetPacket() {
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader(this->ack_header);
    packet->AddHeader(this->network_header);
    //if(this->GetPayloadSize() != 0)
    //    packet->AddTrailer(this->network_payload);
    return packet;
};

uint32_t FrameAckPacket::GetFrameId() {
    return this->ack_header.frame_id;
};

Time FrameAckPacket::GetFrameEncodeTime() {
    return this->ack_header.frame_encode_time;
};


/* class NetStatePacket */
TypeId NetStatePacket::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetStatePacket")
        .SetParent<ControlPacket> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<NetStatePacket> ()
    ;
    return tid;
};

NetStatePacket::NetStatePacket() : ControlPacket(PacketType::NETSTATE_PKT) {
 };

NetStatePacket::~NetStatePacket() {};

NetStatePacket::NetStatePacket(Ptr<Packet> packet): ControlPacket(PacketType::NETSTATE_PKT) {
    // NetworkPacketHeader has been removed in NetworkPacket::ToInstance
    packet->RemoveHeader(this->net_state_header);
    //packet->RemoveTrailer(this->network_payload);
};


Ptr<Packet> NetStatePacket::ToNetPacket() {
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader(this->net_state_header);
    packet->AddHeader(this->network_header);
    //if(this->GetPayloadSize() != 0)
    //    packet->AddTrailer(this->network_payload);
    return packet;
};

Ptr<NetStates> NetStatePacket::GetNetStates() {
    return this->net_state_header.netstates;
};

void NetStatePacket::SetNetStates(Ptr<NetStates> netstate) {
    this->net_state_header.netstates = netstate;
};

} // namespace ns3
