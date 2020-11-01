from run import main
import os

datalist = os.listdir('data')

model_name = 'vanilla_old'

for trial in range(10):
    for data in datalist:
        if data =='.ipynb_checkpoints':
            continue
        print(data, ' learning start!')
        result_16 = main(filename=data ,batch=8, epoch=1000, model_name=model_name)
        # result_16 = main(filename='TwoLeadECG', batch=8, epoch=1000, model_name=model_name)
        # result_16 = main(filename='GunPointAgeSpan' ,batch=8, epoch=1000, model_name=model_name)
        # result_16 = main(filename='Herring' ,batch=8, epoch=1000, model_name=model_name)
        # result_16 = main(filename='ItalyPowerDemand' ,batch=8, epoch=1000, model_name=model_name)
        # result_16 = main(filename='MoteStrain' ,batch=8, epoch=1000, model_name=model_name)
        # break

        # print('for batch 16 : ', result_16)
        # print('for batch 200 : ', result_64)
        file = 'Result/' + model_name + '/'+data+'.txt'
        fw = open(file, 'a')
        fw.write('trial '+str(trial+1)+' for batch 64 : '+ str(result_16)+'\n')
        fw.close()
