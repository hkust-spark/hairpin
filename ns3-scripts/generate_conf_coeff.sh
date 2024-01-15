fname="wifi-coeff.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    conf="rtc-test --vary=1 --trace=traces/${trace}"
    rtx=dupack
    for coeff in 1e-1 1e-2 1e-4 1e-7
    do
        echo '"'${conf}' --fecPolicy=hairpin --rtxPolicy='${rtx}' --coeff='${coeff}'",logs/'${rtx}'hairpin'${coeff}'/'${trace}'/output.log' >> ${fname}
    done
done
