
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "math.h"
#include "webrtc-fec-array.h"
#include "webrtc-adjust-array.h"


// interface
/**
 * @brief Get FEC parameters given network status
 *
 * @param loss packet loss rate, [0.0000, 0.5000]
 * @param group_size Data packet num, [5, 55]
 * @param rtt rtt in ms
 * @param bitrate Video bitrate in Mbps, [2.00, 30.00]
 * @param beta_0 stores FEC rate for the first transmission, [0, 1]
 */
void get_fec_rate_webrtc_rtt(double_t loss, uint8_t group_size, uint16_t rtt, uint16_t bitrate, double_t * fec_rate){
    get_fec_rate_webrtc(loss, group_size, bitrate, fec_rate);
    *fec_rate = (*fec_rate) * (adjust_rtt_array_webrtc[rtt] / 100.0);
};