
#ifndef WEBRTC_ADJUST_ARRAY_H
#define WEBRTC_ADJUST_ARRAY_H


#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "math.h"


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
void get_fec_rate_webrtc_rtt(double_t loss, uint8_t group_size, uint16_t rtt, uint16_t bitrate, double_t * fec_rate);


// Table for adjusting FEC rate for NACK/FEC protection method
// Table values are built as a sigmoid function, ranging from 0 to
// kHighRttNackMs (100), based on the HybridNackTH values defined in
// media_opt_util.h.
/* Copied from nack_fec_tables.h */
static const uint8_t adjust_rtt_array_webrtc[100] = {
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
1,
1,
1,
1,
2,
2,
2,
3,
3,
4,
5,
6,
7,
9,
10,
12,
15,
18,
21,
24,
28,
32,
37,
41,
46,
51,
56,
61,
66,
70,
74,
78,
81,
84,
86,
89,
90,
92,
93,
95,
95,
96,
97,
97,
98,
98,
99,
99,
99,
99,
99,
99,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
100,
};

#endif  // WEBRTC_ADJUST_ARRAY_H
