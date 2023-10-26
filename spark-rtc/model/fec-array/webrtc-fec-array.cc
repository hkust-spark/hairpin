
#include "webrtc-fec-array.h"

void get_fec_rate_webrtc(double_t loss, uint8_t group_size, uint16_t bitrate, double_t * fec_rate) {
    // loss_index = round((loss - start) / interval)
    if(loss < 0.000000)    loss = 0.000000;
    if(loss > 0.500000)    loss = 0.500000;
    uint8_t loss_index = (uint8_t) round((loss - 0.000000) / 0.010000);
    // group_size_index = (group_size - start) / interval
    if(group_size < 5)    group_size = 5;
    if(group_size > 55)    group_size = 55;
    group_size = round(((double_t) group_size - 5) / 5) * 5 + 5;
    uint8_t group_size_index = (uint8_t) round(((double_t) group_size - 5) / 5);
    // bitrate_index = round((bitrate - start) / interval)
    if(bitrate < 2.000000)    bitrate = 2.000000;
    if(bitrate > 30.000000)    bitrate = 30.000000;
    uint8_t bitrate_index = (uint8_t) round((bitrate  - 2.000000) / 1.000000);

    /* array index */
    uint64_t index =
        loss_index * 319
        + group_size_index * 29
        + bitrate_index;
    /* assignment */
    *fec_rate = ((double_t) fec_array_webrtc[index]) / group_size;
};
