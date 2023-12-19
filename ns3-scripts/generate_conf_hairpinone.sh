fname="wifi-hairpinone.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    conf="rtc-test --vary=1 --trace=traces/${trace}"
    rtx=dupack
    for coeff in 1e+01 5e+00 1e+00 1e-01 1e-02 1e-04 1e-07
    do
        echo '"'${conf}' --fecPolicy=hairpinone --rtxPolicy='${rtx}' --coeff='${coeff}'",logs/'${rtx}'hairpinone'${coeff}'/'${trace}'/output.log' >> ${fname}
    done
done
