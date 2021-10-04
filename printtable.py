from data_handling import print_table

urls=['/media/VA/mlruns/64/8e12317985294775bb2b4ad5d87fe553/artifacts/20210910_201254.log.json','/media/VA/mlruns/64/8f453a7e147043cd8558eb29a3938dfd/artifacts/20210913_131729.log.json','/media/VA/mlruns/64/5d31462a1fe54d6c985a16601af49fa9/artifacts/20210913_134605.log.json']


#Prints a table containing info from all the specified runs of the experiment
print_table(urls)