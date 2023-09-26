#include "fec-policy.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("FECPolicy");

/* class FECPolicy */

FECPolicy::FECParam::FECParam(uint16_t group_size, double_t fec_rate) {
    this->fec_group_size = group_size;
    this->fec_rate = fec_rate;
};

FECPolicy::FECParam::FECParam() {};
FECPolicy::FECParam::~FECParam() {};

TypeId FECPolicy::NetStat::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FECPolicy::NetStat")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
        .AddConstructor<FECPolicy::NetStat> ()
    ;
    return tid;
};

FECPolicy::NetStat::NetStat() {};
FECPolicy::NetStat::~NetStat() {};

FECPolicy::FECPolicy(Time rtx_check_intvl) : 
    pacing_flag {false}, 
    rtx_check_interval {rtx_check_intvl},
    max_fec_rate {-1},
    loss_fixed_flag {false},
    fixed_loss {0}
    {};

FECPolicy::~FECPolicy() {};

TypeId FECPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FECPolicy")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

void FECPolicy::SetMaxFECRate(double_t max_fec_rate) {
    this->max_fec_rate = max_fec_rate;
};

void FECPolicy::SetFixedLoss(double_t loss_rate) {
    this->loss_fixed_flag = true;
    this->fixed_loss = loss_rate;
};


bool FECPolicy::GetPacingFlag() {
    return this->pacing_flag;
};


FECPolicy::FECParam FECPolicy::GetFECParam(
    Ptr<NetStat> statistic, uint32_t bitrate,
    uint16_t ddl, uint16_t ddl_left,
    bool is_rtx, uint8_t frame_size,
    uint16_t max_group_size, bool fix_group_size
) {
    if(this->loss_fixed_flag) {
        statistic->curLossRate = this->fixed_loss;
    }
    FECPolicy::FECParam param = this->GetPolicyFECParam(
        statistic, bitrate,
        ddl, ddl_left,
        is_rtx, frame_size,
        max_group_size, fix_group_size
    );
    if (this->max_fec_rate != -1) {
        param.fec_rate = MIN(this->max_fec_rate, param.fec_rate);
    }
    NS_ASSERT(param.fec_rate >= 0);
    return param;
};

}; // namespace ns3