for trace in `ls logs/dupackhairpin7`
do
    awk '{
        if ($7 - $4 > 100) {
            framecnt ++
            timecnt += $7 - tot
        }
        tot = $7
    } END {
        print framecnt / NR, timecnt / tot
    }' logs/dupackhairpin7/${trace}/app.log | grep -v "0 0"
done