fname="test-rtx.conf"
rm ${fname}
for trace in `ls traces-wifi`
do
    conf="rtc-test --vary=1 --trace=traces-wifi/${trace}"
    rtx=dupack
    echo '"'${conf}' --fecPolicy=tokenrtx --rtxPolicy='${rtx}'",logs/'${rtx}'tokenrtx/'${trace}'/output.log' >> ${fname}
    for rate in 0.02 0.05 0.10 0.20 0.30 0.50 1.00 2.00
    do
        echo '"'${conf}' --fecPolicy=fixedrtx --param1='${rate}'",logs/'${rtx}'fixedrtx'${rate}'/'${trace}'/output.log' >> ${fname}
    done
done
