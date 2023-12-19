fname="wifi-lin.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    conf="rtc-test --vary=1 --trace=traces/${trace}"

    rtx=dupack

    for slope in 0.50 1.00 2.00 4.00
    do
        echo '"'${conf}' --fecPolicy=lin --param1='${slope}'",logs/'${rtx}'lin'${slope}'/'${trace}'/output.log' >> ${fname}
    done
done
