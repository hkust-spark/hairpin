#ifndef FEC_POLICY_H
#define FEC_POLICY_H

#include "common-header.h"
#include "packet-group.h"
#include "ns3/object.h"
#include "ns3/nstime.h"
#include <vector>

namespace ns3{

class GameClient;

class FECPolicy : public Object {
public:
    class FECParam{
    public:
        /* FEC group size (num of data packets in a group), should be smaller than a frame */
        uint16_t fec_group_size;
        double_t fec_rate;
        FECParam();
        ~FECParam();
        FECParam(uint16_t fec_group_size, double_t fec_rate);
    };
    // double_t FECRate;
    class NetStat : public Object {
    public:
        static TypeId GetTypeId (void);
        Time curRtt;   /* in ms */
        Time srtt;
        Time minRtt;
        Time rttSd;
        double_t curBw;    /* in Mbps */
        double_t curLossRate;
        std::vector<int> loss_seq;
        Time oneWayDispersion; /*in ms*/
        Time rt_dispersion;     /*in ms*/
        NetStat();
        ~NetStat();
    };
public:
    FECPolicy(Time rtx_check_intvl);
    ~FECPolicy();
    static TypeId GetTypeId (void);
protected:
    Time rtt;
    bool pacing_flag;
    Time rtx_check_interval;
    EventId check_rtx_event;
    GameClient * game_client;
    double_t max_fec_rate;
private:
    bool loss_fixed_flag;
    double_t fixed_loss;
    /**
     * @brief Get FEC parameter given certain group size, called when the best group size is larger than packets in a frame
     *
     * @param statistic Loss, rtt, bandwidth, loss rate and FEC group delay per packet
     * @param bitrate Video encoding bitrate, in kbps
     * @param ddl Total ddl in ms
     * @param ddl_left DDL left in ms
     * @param max_group_size Max group size
     * @param fix_group_size flag for whether the group size can be smaller than max_group_size
     * @return FECParam FEC group size and FEC rate
     */
    virtual FECParam GetPolicyFECParam(
        Ptr<NetStat> statistic, uint32_t bitrate,
        uint16_t ddl, uint16_t ddl_left,
        bool is_rtx, uint8_t frame_size,
        uint16_t max_group_size, bool fix_group_size
    )=0;
public:
    static const uint8_t MAX_GROUP_SIZE = 100;
    void SetMaxFECRate(double_t max_fec_rate);
    void SetFixedLoss(double_t loss_rate);
    bool GetPacingFlag();

    /**
     * @brief Unified interface to call GetPolicyFECParam
     * 
     * @param statistic 
     * @param bitrate 
     * @param ddl 
     * @param ddl_left 
     * @param is_rtx 
     * @param frame_size 
     * @param max_group_size 
     * @param fix_group_size 
     * @return FECParam 
     */
    FECParam GetFECParam(
        Ptr<NetStat> statistic, uint32_t bitrate,
        uint16_t ddl, uint16_t ddl_left,
        bool is_rtx, uint8_t frame_size,
        uint16_t max_group_size, bool fix_group_size
    );
    virtual std::string GetFecName (void) = 0;
};  // class FECPolicy

}; // namespace ns3

#endif /* FEC_POLICY_H */