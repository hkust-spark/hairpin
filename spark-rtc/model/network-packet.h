#ifndef NETWORK_PACKET_H
#define NETWORK_PACKET_H

#include "common-header.h"
#include "network-packet-header.h"
#include "ns3/object.h"
#include "ns3/network-module.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/assert.h"
#include "ns3/nstime.h"

namespace ns3 {

class GroupPacketInfo;

class NetworkPacket : public Object {
public:
    static TypeId GetTypeId (void);
    NetworkPacket(PacketType);
    ~NetworkPacket();

    /**
     * \brief Get a real network packet
     *
     * \return Ptr<Packet> A UDP packet
     */
    virtual Ptr<Packet> ToNetPacket();
    static Ptr<NetworkPacket> ToInstance(Ptr<Packet>);

/* Packet Meta */
protected:
    uint16_t MAX_PACKET_SIZE;
    Time send_time;             /* the moment it's sent, set in PacketSender */
    // TODO: rcv_time should be set by PacketReceiver when received by client @chenjing98
    Time rcv_time;              /* the moment it's received, set in PacketReceiver */
public:
    void SetSendTime(Time);
    void SetRcvTime(Time);
    Time GetSendTime();
    bool IsPacketSent();
    Time GetRcvTime();

/* Header Section */
protected:
    NetworkPacketHeader network_header;
public:
    PacketType GetPacketType();
    void SetPacketType(PacketType);

/* Payload Section */
protected:
    NetworkPacketPayload network_payload;
public:
    static uint16_t GetMaxPayloadSize();
    uint32_t GetPayloadSize();
    uint8_t * GetPayloadPtr();
    void SetPayload(uint8_t *, uint32_t);
};  // class NetworkPacket


class VideoPacket : public NetworkPacket {
public:
    static const uint16_t RTX_FEC_GROUP_ID = -1;
public:
    static TypeId GetTypeId (void);
    VideoPacket(PacketType);
    ~VideoPacket();
    virtual Ptr<Packet> ToNetPacket();

/* Header Section */
protected:
    Time enqueue_time;          /* the moment it's grouped, set in GameServer::StorePackets */

    bool group_info_set_flag;
    bool batch_info_set_flag;
    VideoPacketHeader video_header;

public:
    void SetEncodeTime(Time);
    void SetEnqueueTime(Time);
    Time GetEncodeTime();
    Time GetEnqueueTime();

    void SetFECGroup(uint32_t, uint16_t, uint16_t, uint16_t);
    void SetFECBatch(uint32_t, uint16_t, uint16_t, uint16_t);
    void ClearFECBatch();
    void SetTXCount(uint8_t);
    void IncreTXCount();

    void SetGlobalId(uint16_t);
    uint16_t GetGlobalId();

    bool GetGroupInfoSetFlag();
    uint32_t GetGroupId();
    uint16_t GetGroupDataNum();
    uint16_t GetGroupFECNum();
    uint16_t GetPktIdGroup();

    bool GetBatchInfoSetFlag();
    uint32_t GetBatchId();
    uint16_t GetBatchDataNum();
    uint16_t GetBatchFECNum();
    uint16_t GetPktIdBatch();

    uint8_t GetTXCount();

    static uint16_t GetMaxPayloadSize();
};  // class VideoPacket


class DataPacket : public VideoPacket {
public:
    static TypeId GetTypeId (void);
    /**
     * \brief Construct a new DataPacket object
     *
     * \param frame_id
     * \param frame_pkt_num
     * \param pkt_id_in_frame
     */
    DataPacket(uint32_t frame_id, uint16_t frame_pkt_num, uint16_t pkt_id_in_frame);

    /**
     * @brief Construct a new DataPacket object given a network packet, used by PacketReceiver
     *
     * @param pkt A network packet
     */
    DataPacket(Ptr<Packet> pkt);

    /**
     * @brief Construct a new DataPacket object using DataPktDigest stored in FECPackets
     *
     * @param data_packet_digest digest about data packets stored in FECPackets
     */
    DataPacket(Ptr<DataPktDigest> data_pkt_digest,
        uint32_t group_id, uint16_t group_data_num, uint16_t group_fec_num,
        uint32_t batch_id, uint16_t batch_data_num, uint16_t batch_fec_num
    );

    ~DataPacket();
    Ptr<Packet> ToNetPacket();

protected:
    DataPacketHeader data_header;
    uint16_t m_dataGlobalId;    /* dataGlobalId is used for all data packets, including rtx ones bu excluding fec */
public:
    static uint16_t GetMaxPayloadSize();
    void SetFrameInfo(uint32_t frame_id, uint16_t frame_pkt_num, uint16_t pkt_id_in_frame);
    void SetLastPkt(bool);

    uint32_t GetFrameId();
    uint16_t GetFramePktNum();
    uint16_t GetPktIdFrame();
    bool GetLastPktMark();
    void SetDataGlobalId (uint16_t);
    uint16_t GetDataGlobalId ();

};  // class DataPacket


class DupFECPacket : public DataPacket {
public:
    static TypeId GetTypeId (void);
    DupFECPacket(Ptr<DataPacket>);
    DupFECPacket(uint32_t, uint16_t, uint16_t);
    DupFECPacket(Ptr<Packet>);
    ~DupFECPacket();
};  // class DupFECPacket


class FECPacket : public VideoPacket {
public:
    static TypeId GetTypeId (void);
    FECPacket(uint8_t tx_count, std::vector<Ptr<DataPacket>> data_pkts);
    FECPacket(Ptr<Packet>);
    ~FECPacket();
    Ptr<Packet> ToNetPacket();
protected:
    FECPacketHeader fec_header;
private:
    uint32_t GetHeaderLength();
public:
    std::vector<Ptr<DataPktDigest>> GetDataPacketDigests();
    void SetDataPackets(std::vector<Ptr<DataPacket>> data_pkts);
};  // class FECPacket


class ControlPacket : public NetworkPacket {
public:
    static TypeId GetTypeId (void);
    ControlPacket(PacketType);
    ~ControlPacket();
    virtual Ptr<Packet> ToNetPacket();
}; // class ControlPacket


class AckPacket : public ControlPacket {
public:
    static TypeId GetTypeId (void);
    AckPacket();
    AckPacket(Ptr<Packet>);
    AckPacket(std::vector<Ptr<GroupPacketInfo>>, uint16_t);
    ~AckPacket();
    Ptr<Packet> ToNetPacket();
protected:
    AckPacketHeader ack_header;
public:
    std::vector<Ptr<GroupPacketInfo>> GetAckedPktInfos();
    uint16_t GetLastPktId();
};

class FrameAckPacket : public ControlPacket {
public:
    static TypeId GetTypeId (void);
    FrameAckPacket();
    FrameAckPacket(Ptr<Packet>);
    FrameAckPacket(uint32_t, Time);
    ~FrameAckPacket();
    Ptr<Packet> ToNetPacket();
protected:
    FrameAckPacketHeader ack_header;
public:
    uint32_t GetFrameId();
    Time GetFrameEncodeTime();
};

class NetStatePacket : public ControlPacket {
public:
    static TypeId GetTypeId (void);
    NetStatePacket();
    NetStatePacket(Ptr<Packet>);
    ~NetStatePacket();
    Ptr<Packet> ToNetPacket();
protected:
    NetStatePacketHeader net_state_header;
public:
    Ptr<NetStates> GetNetStates();
    void SetNetStates(Ptr<NetStates>);
};  // class NetStatePacket

};  // namespace ns3

#endif  /* NETWORK_PACKET_H */