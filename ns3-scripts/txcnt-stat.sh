awk '{
    cnt[$7]++
} END{
    for (val in cnt) {
        printf("%d\t%.6f\n", val, (cnt[0]-cnt[val])/cnt[0])
    }
}' logs/$1/wifi-*/fec.log