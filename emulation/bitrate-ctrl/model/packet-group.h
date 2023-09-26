#ifndef PACKET_GROUP_H
#define PACKET_GROUP_H

#include "common-header.h"
#include "network-packet.h"
#include "ns3/simulator.h"
#include <unordered_set>
#include <unordered_map>

namespace ns3 {
class PacketBatch;

class PacketGroup : public Object {
public:
    static TypeId GetTypeId (void);

    /**
     * @brief Construct a new PacketGroup object with no packets in this group
     *
     * @param group_id Group ID
     * @param data_num The num of data packets in this group
     * @param encode_time Frame encode time.
     */
    PacketGroup(uint32_t group_id, uint16_t data_num, Time encode_time);
    /**
     * @brief Construct a new PacketGroup object with the first packet received in this group
     *
     * @param pkt The first packet received of this group
     * @param group_delay Packet interval.
     */
    PacketGroup(Ptr<VideoPacket> pkt, Time group_delay);
    ~PacketGroup();
    std::unordered_set<uint16_t> decoded_pkts;
private:
    uint32_t group_id;

    uint16_t data_num;
    uint16_t fec_num;

    // Assume all packets comes from the same frame
    Time encode_time;
    Time first_rcv_time;
    Time last_rcv_time_tx_0;
    Time group_delay;

    uint16_t pkt_cnt_tx_0;

    std::vector<Ptr<DataPacket>> undecoded_pkts;
    std::unordered_map<uint32_t, Ptr<PacketBatch>> incomplete_batches;
    std::unordered_map<uint32_t, Ptr<PacketBatch>> complete_batches;

    // mark the next round of rtx packets
    uint8_t next_rtx_count;

    // statistics
    uint16_t last_pkt_id_tx_0;
    uint8_t max_tx_count;
    uint16_t first_tx_data_count;
    uint16_t first_tx_fec_count;
    uint16_t rtx_data_count;
    uint16_t rtx_fec_count;
    std::unordered_map<uint8_t, uint32_t> pkt_arrive_dis;

public:
    uint32_t GetGroupId();
    uint16_t GetDataNum();
    uint16_t GetFECNum();
    Time GetEncodeTime();
    Time GetLastRcvTimeTx0();
    Time GetAvgPktInterval();
    uint8_t GetMaxTxCount();
    uint16_t GetFirstTxDataCount();
    uint16_t GetFirstTxFECCount();
    uint16_t GetRtxDataCount();
    uint16_t GetRtxFECCount();
    std::unordered_map<uint8_t, uint32_t> GetPktArriveDistribution();

public:
    void InitGroup(Ptr<VideoPacket> pkt);
    void AddPacket(Ptr<VideoPacket> pkt, Time group_delay);
    std::vector<Ptr<DataPacket>> GetUndecodedPackets();
    bool CheckComplete();
private:
    void UpdateExpectedCompletionTime(Ptr<VideoPacket> pkt, Time group_delay);
};  // class PacketGroup


class PacketBatch : public Object {
public:
    static TypeId GetTypeId (void);
    PacketBatch(Ptr<VideoPacket> pkt, Ptr<PacketGroup> packet_group);
    ~PacketBatch();

private:
     Ptr<PacketGroup> packet_group;

    uint32_t batch_id;
    uint16_t data_num;
    uint16_t fec_num;

    std::unordered_set<uint16_t> decoded_pkts;
    std::vector<Ptr<DataPacket>> undecoded_pkts;

    std::unordered_map<uint16_t, Ptr<DataPacket>> data_pkts;
    std::unordered_map<uint16_t, Ptr<FECPacket>> fec_pkts;

public:
    void AddPacket(Ptr<VideoPacket> pkt);
    bool CheckComplete();
    std::vector<Ptr<DataPacket>> GetUndecodedPackets();
};  // class PacketBatch


}; // namespace ns3

#endif  /* PACKET_GROUP_H */