#ifndef NETWORK_PACKET_HEADER_H
#define NETWORK_PACKET_HEADER_H

#include "common-header.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"
#include "ns3/assert.h"
#include "ns3/nstime.h"
#include "ns3/object.h"

namespace ns3 {

class DataPacket;

class GroupPacketInfo : public Object {
public:
    enum PacketState {INFLIGHT, RCVD_PREV_DATA};
    static TypeId GetTypeId (void);
    uint32_t m_groupId;
    uint16_t m_pktIdInGroup;
    uint16_t m_dataGlobalId;
    uint16_t m_globalId;
    uint8_t m_txCnt;
    PacketState m_state;
    GroupPacketInfo ();
    GroupPacketInfo (uint32_t group_id, uint16_t pktIdInGroup, uint16_t dataGlobalId, uint16_t globalId, 
        uint8_t txCnt = 0);
    ~GroupPacketInfo ();
};

enum PacketType { DATA_PKT, DUP_FEC_PKT, FEC_PKT, RTX_REQ_PKT, ACK_PKT, FRAME_ACK_PKT, NETSTATE_PKT };

class NetworkPacketHeader : public SimpleRefCount<NetworkPacketHeader,Header> {
private:
    PacketType packet_type;
public:
    friend class NetworkPacket;
    static TypeId GetTypeId (void);
    NetworkPacketHeader();
    ~NetworkPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
};  // class NetworkPacketHeader

class NetworkPacketPayload : public SimpleRefCount<NetworkPacketPayload,Trailer> {
protected:
    uint8_t * payload_buffer;
    uint32_t payload_size;
public:
    friend class NetworkPacket;
    static TypeId GetTypeId (void);
    NetworkPacketPayload();
    ~NetworkPacketPayload();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator end);
    uint32_t Deserialize (Buffer::Iterator start, Buffer::Iterator end);
    void Print (std::ostream &os) const;
};

class VideoPacketHeader : public SimpleRefCount<VideoPacketHeader,Header> {
/* Header Section */
private:
    Time encode_time;           /* the moment it's finished encoding, set in GameServer::SendFrame */

    uint16_t global_id;         /* Global id that each packet sent from Server has */

    uint32_t group_id;          /* The id of packet group the packet is in */
    uint16_t group_data_num;    /* The num of data packets in the group initially */
    uint16_t group_fec_num;     /* The num of FEC packets in the group initially */
    uint16_t pkt_id_in_group;

    uint32_t batch_id;          /* The id of packet batch the packet is in */
    uint16_t batch_data_num;
    uint16_t batch_fec_num;
    uint16_t pkt_id_in_batch;

    uint8_t tx_count;           /* The num of transmissions of this packet */
public:
    friend class VideoPacket;
    static TypeId GetTypeId (void);
    VideoPacketHeader();
    ~VideoPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
};  // class VideoPacketHeader

class DataPktFrameInfo : public Object {
public:
    static TypeId GetTypeId (void);
    uint32_t frame_id;
    uint16_t pkt_id_in_frame;
    DataPktFrameInfo(uint32_t, uint16_t);
    ~DataPktFrameInfo();
};


class DataPktDigest : public Object {
public:
    static TypeId GetTypeId (void);
    uint16_t pkt_id_in_batch;
    uint16_t pkt_id_in_group;
    uint32_t frame_id;
    uint16_t frame_pkt_num;
    uint16_t pkt_id_in_frame;
    DataPktDigest();
    DataPktDigest(Ptr<DataPacket> pkt);
    ~DataPktDigest();
};

class DataPacketHeader : public SimpleRefCount<DataPacketHeader,Header> {
private:
    uint32_t frame_id;          /* Id of the frame that the payload of the packet is consisted of */
    uint16_t frame_pkt_num;     /* The num of data packets containing data from the frame */
    uint16_t pkt_id_in_frame;
    bool last_pkt_mark;         /* If it's the last packet of the frame, defualt false */
public:
    friend class DataPacket;
    static TypeId GetTypeId (void);
    DataPacketHeader();
    ~DataPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
};  // class DataPacketHeader

class FECPacketHeader : public SimpleRefCount<FECPacketHeader,Header> {
private:
    std::vector<Ptr<DataPktDigest>> data_pkts;
public:
    friend class FECPacket;
    static TypeId GetTypeId (void);
    FECPacketHeader();
    ~FECPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
};  // class FECPacket

class RtxRequestPacketHeader : public SimpleRefCount<RtxRequestPacketHeader,Header> {
private:
    uint8_t frame_req; // request retransmission of a whole frame or group
    uint32_t rtx_frame_id;
    std::vector<Ptr<GroupPacketInfo>> pkt_infos;
public:
    friend class RtxRequestPacket;
    static TypeId GetTypeId (void);
    RtxRequestPacketHeader();
    ~RtxRequestPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
}; // class RtxRequestPacketHeader

class AckPacketHeader : public SimpleRefCount<AckPacketHeader,Header> {
private:
    std::vector<Ptr<GroupPacketInfo>> pkt_infos;
    uint16_t last_pkt_id;
public:
    friend class AckPacket;
    static TypeId GetTypeId (void);
    AckPacketHeader();
    ~AckPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
}; // class AckPacketHeader

class FrameAckPacketHeader : public SimpleRefCount<FrameAckPacketHeader,Header> {
private:
    uint32_t frame_id;
    Time frame_encode_time;
public:
    friend class FrameAckPacket;
    static TypeId GetTypeId (void);
    FrameAckPacketHeader();
    ~FrameAckPacketHeader();
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
}; // class FrameAckPacketHeader

class RcvTime : public Object {
public:
    RcvTime(uint32_t pkt_id_, uint32_t rt_us_, uint32_t pkt_size_) {
        this->pkt_id = pkt_id_;
        this->rt_us = rt_us_;
        this->pkt_size = pkt_size_;
    };
    ~RcvTime() {};
    uint32_t pkt_id;
    uint32_t rt_us;
    uint32_t pkt_size;
}; // class RcvTime

class NetStates : public Object {
public:
    static TypeId GetTypeId (void);
    double_t loss_rate;
    uint32_t throughput_kbps;
    uint16_t fec_group_delay_us;
    std::vector<int> loss_seq;
    std::vector<Ptr<RcvTime>> recvtime_hist;
    NetStates();
    NetStates(double_t lr, uint32_t tp, uint16_t gd);
    ~NetStates();
}; // class NetStates


class NetStatePacketHeader : public SimpleRefCount<NetStatePacketHeader,Header> {
private:
    Ptr<NetStates> netstates;
public:
    friend class NetStatePacket;
    static TypeId GetTypeId (void);
    TypeId GetInstanceTypeId (void) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start);
    void Print (std::ostream &os) const;
    NetStatePacketHeader();
    ~NetStatePacketHeader();

};  // class NetStatePacketHeader

};  // namespace ns3

#endif  /* NETWORK_PACKET_HEADER_H */