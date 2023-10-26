#include "hairpin-policy.h"
#include <fstream>

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("Hairpin");

/* class HairpinPolicy */
const std::vector<uint8_t> HairpinPolicy::group_size_list = {5, 15, 20, 25, 30, 35, 40, 45, 50, 55};

HairpinPolicy::HairpinPolicy (uint16_t delayDdl, double_t qoeCoeff, bool isRtx, bool isCap)
: FECPolicy(MilliSeconds(1)) 
, k_betaArraySize {462825}
, k_blockArraySize {7209972}
, k_paramDir {"../../../simulation/code/model/"} 
, m_isBlockSizeOpt {false} {
    k_qoeCoeff = qoeCoeff;
    k_delayDdl = delayDdl;
    this->pacing_flag = false;

    char buf[100];
    snprintf (buf, sizeof (buf), "array-rtx%d-cap%d-coeff%.0e", isRtx, isCap, k_qoeCoeff);
    std::string dataConf = buf;
    m_betaArray = new uint8_t[k_betaArraySize];
    std::ifstream betaIo (k_paramDir + "beta-" + dataConf + ".bin", std::ifstream::binary);
    if (betaIo.fail ())
        NS_FATAL_ERROR ("Cannot open " + k_paramDir + "beta-" + dataConf + ".bin");
    betaIo.seekg (0, std::ios::beg);
    betaIo.read ((char *) m_betaArray, sizeof (uint8_t) * k_betaArraySize);
    betaIo.close ();
    NS_LOG_INFO ("Read beta array from " + k_paramDir + "beta-" + dataConf + ".bin");
    if (m_isBlockSizeOpt) {
        m_blockArray = new uint8_t[k_blockArraySize];
        std::ifstream blockIo (k_paramDir + "block-" + dataConf + ".bin", std::ifstream::binary);
        NS_ASSERT (!blockIo.fail ());
        blockIo.seekg (0, std::ios::beg);
        blockIo.read ((char *) m_blockArray, sizeof (uint8_t) * k_blockArraySize);
        blockIo.close ();
        NS_LOG_INFO ("Read block array from " + k_paramDir + "block-" + dataConf + ".bin");
    }
};

HairpinPolicy::HairpinPolicy() : FECPolicy(MilliSeconds(1)) {};
HairpinPolicy::~HairpinPolicy() {};

TypeId HairpinPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::HairpinPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<HairpinPolicy> ()
    ;
    return tid;
};

uint8_t HairpinPolicy::GetFecCnt (double_t loss, uint8_t frameSize, uint16_t remainingTime, uint16_t rtt, uint8_t packet) {
        // loss_index = round((loss - start) / interval)
    loss = (loss < 0.00) ? 0.00 : ((loss > 0.50) ? 0.50 : loss);
    uint8_t loss_index = (uint8_t) round((loss - 0.00) / 0.01);

    // frame_size_index = ceil((frame_size - start) / interval)
    frameSize = (frameSize < 5) ? 5 : ((frameSize > 55) ? 55 : frameSize);
    uint8_t frame_size_index = (uint8_t) ceil(((double_t) frameSize - 5) / 5);

    // compute the remaining layer
    uint8_t layer = (uint8_t) (remainingTime / rtt);
    layer = (layer < 1) ? 1 : ((layer > 15) ? 15 : layer);
    uint8_t layer_index = (uint8_t) round(((double_t) layer - 1) / 1);

    // For hairpinone baseline
    if (k_delayDdl == 0)
        layer_index = 0;

    // packet_index = round((packet - start) / interval)
    packet = (packet < 1) ? 1 : ((packet > 55) ? 55 : packet);
    uint8_t packet_index = (uint8_t) round(((double_t) packet - 1) / 1);

    /* array index */
    uint64_t index =
        loss_index * 9075
        + frame_size_index * 825
        + layer_index * 55
        + packet_index;

    /* assignment */
    return m_betaArray[index];
}

uint8_t HairpinPolicy::GetBlockSize (double_t loss, uint8_t frame_size, uint16_t ddl, uint16_t rtt, double_t rdisp) {
    // loss_index = round((loss - start) / interval)
    loss = (loss < 0.00) ? 0.00 : ((loss > 0.50) ? 0.50 : loss);
    uint8_t loss_index = (uint8_t) round((loss - 0.00) / 0.01);

    // frame_size_index = ceil((frame_size - start) / interval)
    frame_size = (frame_size < 5) ? 5 : ((frame_size > 55) ? 55 : frame_size);
    uint8_t frame_size_index = (uint8_t) ceil(((double_t) frame_size - 5) / 5);

    // ddl_index = round((ddl - start) / interval)
    ddl = (ddl < 20) ? 20 : ((ddl > 140) ? 140 : ddl);
    uint8_t ddl_index = (uint8_t) round(((double_t) ddl - 20) / 20);

    // rtt_index = round((rtt - start) / interval)
    rtt = (rtt < 10) ? 10 : ((rtt > 80) ? 80 : rtt);
    uint8_t rtt_index = (uint8_t) round(((double_t) rtt - 10) / 2);

    // rdisp_index = round((rdisp - start) / interval)
    rdisp = (rdisp < 0.00) ? 0.00 : ((rdisp > 1.00) ? 1.00 : rdisp);
    uint8_t rdisp_index = (uint8_t) round((rdisp - 0.00) / 0.02);

    /* array index */
    uint64_t index =
        loss_index * 141372
        + frame_size_index * 12852
        + ddl_index * 1836
        + rtt_index * 51
        + rdisp_index;
    /* assignment */
    return m_blockArray[index];
}

FECPolicy::FECParam HairpinPolicy::GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize
) {
    uint8_t fecCount = 0;
    uint8_t blockSize = maxGroupSize;

    if (m_isRtx || !isRtx) {
        // Block size optimization
        // rtt = srtt + 1 * std
        uint16_t rtt = (uint16_t) ceil ((statistic->srtt + 1 * statistic->rttSd).GetMilliSeconds ());
        ddlLeft = std::max (0, int (ddlLeft) - int (rtt));
        if (m_isBlockSizeOpt && (!fixGroupSize && frameSize == maxGroupSize)) {
            // group size is not fixed
            blockSize = GetBlockSize (statistic->curLossRate, frameSize,
                ddl, rtt, statistic->oneWayDispersion.GetMicroSeconds () / 1e3);
        }
        fecCount = GetFecCnt (statistic->curLossRate, frameSize, ddlLeft, rtt, blockSize);
        NS_ASSERT (fecCount < 255);
    }
    double_t fecRate = ((double_t) fecCount) / blockSize;
    return FECParam (blockSize, fecRate);
};

std::string HairpinPolicy::GetFecName (void) {
    return "HairpinPolicy";
}

}; // namespace ns3