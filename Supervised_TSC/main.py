from run_multi import main
import os

datalist = os.listdir('data')

for trial in range(10):
    for data in datalist:
        if data =='.ipynb_checkpoints':
            continue
        print(data, ' learning start!')
        result_16 = main(filename=data ,batch=64, epoch=1500)

        # print('for batch 16 : ', result_16)
        # print('for batch 200 : ', result_64)
        file = 'Result/sunday/'+data+'.txt'
        fw = open(file, 'a')
        fw.write('trial '+str(trial+1)+' for batch 64 : '+ str(result_16)+'\n')
        fw.close()
