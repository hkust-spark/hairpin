#include "network-packet-header.h"


namespace ns3 {
NS_LOG_COMPONENT_DEFINE("PacketHeader");

TypeId GroupPacketInfo::GetTypeId() {
    static TypeId tid = TypeId ("ns3::GroupPacketInfo")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

GroupPacketInfo::GroupPacketInfo () {};

GroupPacketInfo::GroupPacketInfo (uint32_t groupId, uint16_t pktIdInGroup, uint16_t dataGlobalId, uint16_t globalId,
    uint8_t txCnt) {
    m_groupId = groupId;
    m_pktIdInGroup = pktIdInGroup;
    m_dataGlobalId = dataGlobalId;
    m_globalId = globalId;
    m_txCnt = txCnt;
    m_state = INFLIGHT;
};

GroupPacketInfo::~GroupPacketInfo() {};

// class NetworkPacketHeader
TypeId NetworkPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetworkPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<NetworkPacketHeader> ()
    ;
    return tid;
};

NetworkPacketHeader::NetworkPacketHeader() {};
NetworkPacketHeader::~NetworkPacketHeader() {};

TypeId NetworkPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t NetworkPacketHeader::GetSerializedSize () const {
    return sizeof(PacketType);
};

void NetworkPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU32((uint32_t)this->packet_type);
};

uint32_t NetworkPacketHeader::Deserialize (Buffer::Iterator start) {
    this->packet_type = static_cast<PacketType> (start.ReadNtohU32());
    return 4;
};

void NetworkPacketHeader::Print (std::ostream &os) const {
    os << "PacketType=" << this->packet_type;
};

// class NetworkPacketPayload
TypeId NetworkPacketPayload::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetworkPacketPayload")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

NetworkPacketPayload::NetworkPacketPayload() {
    this->payload_size = 0;
    this->payload_buffer = new uint8_t[1];
};

NetworkPacketPayload::~NetworkPacketPayload() {};
TypeId NetworkPacketPayload::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t NetworkPacketPayload::GetSerializedSize () const {
    return this->payload_size;
};

void NetworkPacketPayload::Serialize (Buffer::Iterator end) const {
    while(!end.IsStart()) end.Prev();
    for(uint32_t i = 0;i < this->payload_size;i++) {
        end.WriteU8(this->payload_buffer[i]);
    }
};

uint32_t NetworkPacketPayload::Deserialize (Buffer::Iterator end) {
    auto start = end;
    while(!start.IsStart()) start.Prev();
    return this->Deserialize(start, end);
};

uint32_t NetworkPacketPayload::Deserialize (Buffer::Iterator start, Buffer::Iterator end) {
    this->payload_size = end.GetDistanceFrom(start);
    this->payload_buffer = new uint8_t[this->payload_size];
    for(uint32_t i = 0;i < this->payload_size;i++) {
        this->payload_buffer[i] = start.ReadU8();
    }
    return this->payload_size;
};

void NetworkPacketPayload::Print (std::ostream &os) const {
    os << "PayloadSize=" << this->payload_size;
};

// class VideoPacketHeader
TypeId VideoPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::VideoPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<VideoPacketHeader> ()
    ;
    return tid;
};

VideoPacketHeader::VideoPacketHeader() {};
VideoPacketHeader::~VideoPacketHeader() {};

TypeId VideoPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t VideoPacketHeader::GetSerializedSize () const {
    return sizeof(uint64_t) + sizeof(global_id)
        + sizeof(group_id) + sizeof(group_data_num) + sizeof(group_fec_num) + sizeof(pkt_id_in_group)
        + sizeof(batch_id) + sizeof(batch_data_num) + sizeof(batch_fec_num) + sizeof(pkt_id_in_batch)
        + sizeof(tx_count);
};

void VideoPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU64(this->encode_time.GetMilliSeconds());
    start.WriteHtonU16(this->global_id);
    start.WriteHtonU32(this->group_id);
    start.WriteHtonU16(this->group_data_num);
    start.WriteHtonU16(this->group_fec_num);
    start.WriteHtonU16(this->pkt_id_in_group);
    start.WriteHtonU32(this->batch_id);
    start.WriteHtonU16(this->batch_data_num);
    start.WriteHtonU16(this->batch_fec_num);
    start.WriteHtonU16(this->pkt_id_in_batch);
    start.WriteU8(this->tx_count);
};

uint32_t VideoPacketHeader::Deserialize (Buffer::Iterator start) {
    this->encode_time = MilliSeconds(start.ReadNtohU64());
    this->global_id = start.ReadNtohU16();
    this->group_id = start.ReadNtohU32();
    this->group_data_num = start.ReadNtohU16();
    this->group_fec_num = start.ReadNtohU16();
    this->pkt_id_in_group = start.ReadNtohU16();
    this->batch_id = start.ReadNtohU32();
    this->batch_data_num = start.ReadNtohU16();
    this->batch_fec_num = start.ReadNtohU16();
    this->pkt_id_in_batch = start.ReadNtohU16();
    this->tx_count = start.ReadU8();
    return sizeof(uint64_t) + sizeof(global_id)
        + sizeof(group_id) + sizeof(group_data_num) + sizeof(group_fec_num) + sizeof(pkt_id_in_group)
        + sizeof(batch_id) + sizeof(batch_data_num) + sizeof(batch_fec_num) + sizeof(pkt_id_in_batch)
        + sizeof(tx_count);
};

void VideoPacketHeader::Print (std::ostream &os) const {
    os << "this->encode_time" << this->encode_time.GetMilliSeconds();
    os << "this->global_id" << this->global_id;
    os << "this->group_id" << this->group_id;
    os << "this->group_data_num" << this->group_data_num;
    os << "this->group_fec_num" << this->group_fec_num;
    os << "this->pkt_id_in_group" << this->pkt_id_in_group;
    os << "this->batch_id" << this->batch_id;
    os << "this->batch_data_num" << this->batch_data_num;
    os << "this->batch_fec_num" << this->batch_fec_num;
    os << "this->pkt_id_in_batch" << this->pkt_id_in_batch;
};


/* class DataPktFrameInfo */
TypeId DataPktFrameInfo::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DataPktFrameInfo")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

DataPktFrameInfo::DataPktFrameInfo(uint32_t frame_id, uint16_t pkt_id_in_frame) {
    this->frame_id = frame_id;
    this->pkt_id_in_frame = pkt_id_in_frame;
};

DataPktFrameInfo::~DataPktFrameInfo() {};


/* class DataPktDigest */
TypeId DataPktDigest::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DataPktDigest")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

DataPktDigest::~DataPktDigest() {};


// class DataPacketHeader
TypeId DataPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DataPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<DataPacketHeader> ()
    ;
    return tid;
};

DataPacketHeader::DataPacketHeader() {};
DataPacketHeader::~DataPacketHeader() {};

TypeId DataPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t DataPacketHeader::GetSerializedSize () const {
    return sizeof(frame_id) + sizeof(frame_pkt_num) + sizeof(pkt_id_in_frame);
};

void DataPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU32(this->frame_id);
    start.WriteHtonU16(this->frame_pkt_num);
    start.WriteHtonU16(this->pkt_id_in_frame);
};

uint32_t DataPacketHeader::Deserialize (Buffer::Iterator start) {
    this->frame_id = start.ReadNtohU32();
    this->frame_pkt_num = start.ReadNtohU16();
    this->pkt_id_in_frame = start.ReadNtohU16();
    return sizeof(frame_id) + sizeof(frame_pkt_num) + sizeof(pkt_id_in_frame);
};

void DataPacketHeader::Print (std::ostream &os) const {
    os << "this->frame_id" << this->frame_id;
    os << "this->frame_pkt_num" << this->frame_pkt_num;
    os << "this->pkt_id_in_frame" << this->pkt_id_in_frame;
};

// class FECPacketHeader
TypeId FECPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FECPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<FECPacketHeader> ()
    ;
    return tid;
};

FECPacketHeader::FECPacketHeader() {};
FECPacketHeader::~FECPacketHeader() {};

TypeId FECPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t FECPacketHeader::GetSerializedSize () const {
    return this->data_pkts.size() * (2 + 2 + 4 + 2 + 2) + 2;
};

void FECPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU16(this->data_pkts.size());
    for(auto digest : this->data_pkts) {
        start.WriteHtonU16(digest->pkt_id_in_batch);
        start.WriteHtonU16(digest->pkt_id_in_group);
        start.WriteHtonU32(digest->frame_id);
        start.WriteHtonU16(digest->frame_pkt_num);
        start.WriteHtonU16(digest->pkt_id_in_frame);
    }
};

uint32_t FECPacketHeader::Deserialize (Buffer::Iterator start) {
    uint32_t read_size = 2;
    uint16_t pkts_size = start.ReadNtohU16();
    while(pkts_size>0) {
        Ptr<DataPktDigest> digest = Create<DataPktDigest> ();
        digest->pkt_id_in_batch = start.ReadNtohU16();
        digest->pkt_id_in_group = start.ReadNtohU16();
        digest->frame_id = start.ReadNtohU32();
        digest->frame_pkt_num = start.ReadNtohU16();
        digest->pkt_id_in_frame = start.ReadNtohU16();
        this->data_pkts.push_back(digest);
        read_size += 12;
        pkts_size--;
    }
    return read_size;
};

void FECPacketHeader::Print (std::ostream &os) const {
    os << "FEC Group Size=" << this->data_pkts.size();
    for(auto digest : this->data_pkts) {
        os << "digest->pkt_id_in_batch" << digest->pkt_id_in_batch;
        os << "digest->pkt_id_in_group" << digest->pkt_id_in_group;
        os << "digest->frame_id" << digest->frame_id;
        os << "digest->frame_pkt_num" << digest->frame_pkt_num;
        os << "digest->pkt_id_in_frame" << digest->pkt_id_in_frame;
    }
};


// class AckPacketHeader
TypeId AckPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::AckPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<AckPacketHeader> ()
    ;
    return tid;
};

AckPacketHeader::AckPacketHeader() {};
AckPacketHeader::~AckPacketHeader() {};

TypeId AckPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t AckPacketHeader::GetSerializedSize () const {
    return sizeof(uint32_t) +
        this->pkt_infos.size() * 6 /* we use part of GroupPacketInfo */ +
        sizeof(this->last_pkt_id);
};

void AckPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU32(this->pkt_infos.size());
    for(auto pkt_info : this->pkt_infos) {
        start.WriteHtonU32 (pkt_info->m_groupId);
        start.WriteHtonU16 (pkt_info->m_pktIdInGroup);
    }
    start.WriteHtonU16(this->last_pkt_id);
};

uint32_t AckPacketHeader::Deserialize (Buffer::Iterator start) {
    uint32_t read_size = 0;

    // read pkt_infos
    uint32_t pktInfoCnt = start.ReadNtohU32();
    read_size += 4;
    for (uint32_t i = 0; i < pktInfoCnt; i++) {
        Ptr<GroupPacketInfo> pkt = Create<GroupPacketInfo> ();
        pkt->m_groupId = start.ReadNtohU32();
        pkt->m_pktIdInGroup = start.ReadNtohU16();
        this->pkt_infos.push_back(pkt);
        read_size += 6;
    }
    this->last_pkt_id = start.ReadNtohU16();
    for (auto pktInfo : this->pkt_infos) {
        /* this is obviously wrong when there are multiple packets acked in one ackpacket
           I just do not want to reconstruct all the code here */
        pktInfo->m_globalId = last_pkt_id;
    }
    read_size += 2;

    return read_size;
};

void AckPacketHeader::Print (std::ostream &os) const {
    os << "Ack " << this->pkt_infos.size() << "packets, global id: " << this->last_pkt_id;
};

// class FrameAckPacketHeader
TypeId FrameAckPacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FrameAckPacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<FrameAckPacketHeader> ()
    ;
    return tid;
};

FrameAckPacketHeader::FrameAckPacketHeader() {};
FrameAckPacketHeader::~FrameAckPacketHeader() {};

TypeId FrameAckPacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t FrameAckPacketHeader::GetSerializedSize () const {
    return sizeof(uint32_t) + sizeof(uint64_t);
};

void FrameAckPacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU32(this->frame_id);
    start.WriteHtonU64(this->frame_encode_time.GetMicroSeconds());
};

uint32_t FrameAckPacketHeader::Deserialize (Buffer::Iterator start) {
    this->frame_id = start.ReadNtohU32();
    this->frame_encode_time = MicroSeconds(start.ReadNtohU64());
    return sizeof(uint32_t) + sizeof(uint64_t);
};

void FrameAckPacketHeader::Print (std::ostream &os) const {
    os << "ACK for frame" << this->frame_id << " encoded at " << this->frame_encode_time.GetMilliSeconds() << " ms";
};

// class NetStates
TypeId NetStates::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetStates")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

NetStates::NetStates(double_t lr, uint32_t tp, uint16_t gd) {
    this->loss_rate = lr;
    this->throughput_kbps = tp;
    this->fec_group_delay_us = gd;
};

NetStates::NetStates() {};
NetStates::~NetStates() {};

// class NetStatePacketHeader
TypeId NetStatePacketHeader::GetTypeId() {
    static TypeId tid = TypeId ("ns3::NetStatePacketHeader")
        .SetParent<Header> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<NetStatePacketHeader> ()
    ;
    return tid;
};

NetStatePacketHeader::NetStatePacketHeader() {};
NetStatePacketHeader::~NetStatePacketHeader() {};

TypeId NetStatePacketHeader::GetInstanceTypeId (void) const { return GetTypeId(); };

uint32_t NetStatePacketHeader::GetSerializedSize () const {
    return 12 + 4 * netstates->loss_seq.size() + 8 * netstates->recvtime_hist.size();
};

void NetStatePacketHeader::Serialize (Buffer::Iterator start) const {
    start.WriteHtonU16((uint16_t)(this->netstates->loss_rate * 10000));
    start.WriteHtonU32(this->netstates->throughput_kbps);
    start.WriteHtonU16(this->netstates->fec_group_delay_us);

    start.WriteHtonU16(this->netstates->loss_seq.size());
    for(auto loss : this->netstates->loss_seq) {
        if(loss<0){
            start.WriteHtonU16(0);
            start.WriteHtonU16((uint16_t)(-loss));
        }
        else {
            start.WriteHtonU16(1);
            start.WriteHtonU16((uint16_t)loss);
        }
    }
    start.WriteHtonU16(this->netstates->recvtime_hist.size());
    for(auto recvtime : this->netstates->recvtime_hist) {
        start.WriteHtonU32(recvtime->pkt_id);
        start.WriteHtonU32(recvtime->rt_us);
    }
};

uint32_t NetStatePacketHeader::Deserialize (Buffer::Iterator start) {
    Ptr<NetStates> netstate = Create<NetStates> ();
    netstate->loss_rate = (float)(start.ReadNtohU16()) / 10000.;
    netstate->throughput_kbps = start.ReadNtohU32();
    netstate->fec_group_delay_us = start.ReadNtohU16();
    uint32_t read_size = 12;

    uint16_t loss_size = start.ReadNtohU16();
    netstate->loss_seq.clear();
    while (loss_size>0)
    {
        uint16_t sig = start.ReadNtohU16();
        uint16_t value = start.ReadNtohU16();
        if(sig>0) {netstate->loss_seq.push_back(int(value));}
        else {netstate->loss_seq.push_back(-(int)value);}
        read_size += 4;
        loss_size--;
    }

    uint16_t hist_size = start.ReadNtohU16();
    netstate->recvtime_hist.clear();
    while(hist_size>0) {
        uint32_t pkt_id = start.ReadNtohU32();
        uint32_t pkt_rcvtime = start.ReadNtohU32();
        Ptr<RcvTime> rt = Create<RcvTime>(pkt_id, pkt_rcvtime, 0);
        netstate->recvtime_hist.push_back(rt);
        read_size += 8;
        hist_size--;
    }
    this->netstates = netstate;
    return read_size;
};

void NetStatePacketHeader::Print (std::ostream &os) const {
    os << "LossRate=" << this->netstates->loss_rate;
    os << "Throughput(kbps)=" <<this->netstates->throughput_kbps;
    os << "FEC group delay(us)=" << this->netstates->fec_group_delay_us;
    os << "Loss sequence includes " << this->netstates->loss_seq.size() << " consecutive items";
    for(auto loss : this->netstates->loss_seq) {
        if(loss<0) {os << "Loss=" << loss;}
        else {os << "Receive=" <<loss;}
    }
    os << "Recvtime history samples " << this->netstates->recvtime_hist.size() << " packets";
    for(auto recvtime : this->netstates->recvtime_hist) {
        os << "Packet ID=" << recvtime->pkt_id;
        os << "Receive time(us)=" << recvtime->rt_us;
    }
};

};