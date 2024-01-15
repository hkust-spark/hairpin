for alg in dupackawebrtc dupackbolot dupackhairpin7 dupackrtx dupackusf dupackwebrtc
do
    for round in 0 1 2 3 4 5
    do
        echo > frame-loss-${alg}-round-${round}.tmp
        for trace in `ls logs/${alg}`
        do
            awk '{
                if ($7 == "'${round}'") {
                    denom[$3] = $11
                }

                if ($7 == "'${round}'" + 1) {
                    numer[$3] = $11
                }
            } END {
                for (i in denom) {
                    if (i in numer) {
                        print numer[i] / denom[i]
                    } else {
                        print 0
                    }
                }
            }' logs/${alg}/${trace}/fec.log >> frame-loss-${alg}-round-${round}.tmp
        done
        python writecdf.py --input frame-loss-${alg}-round-${round}.tmp --output frame-loss-${alg}-round-${round}.cdf
        awk '{sum+=$1} END {print "'${round}'", sum/NR}' frame-loss-${alg}-round-${round}.tmp >> frame-loss-${alg}-round.stat
        rm frame-loss-${alg}-round-${round}.tmp
    done
done