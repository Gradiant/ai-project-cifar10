

def JSON_datahandling(url):  #Clean the data from the .log.json file
    import json
    with open(url) as file:
        lines= [line.strip() for line in file]
        file.close()
    str1=lines[-1]
    str2=lines[-2] 
    str3=lines[1]
    final1=json.loads(str1)
    final2=json.loads(str2)
    final3=json.loads(str3)

    final={'num_epochs':final1['epoch'],'learning rate':final3['lr'],'loss':final2['loss'],'accuracy_top-1':final1['accuracy_top-1'],'accuracy_top-5':final1['accuracy_top-5']}
    return(final)


def Result_table_gen(urls):  # Combines multiple jsons containing the experiment data
    import pandas as pd
    j=0
    for i in urls:
        if j>0 :
            data=JSON_datahandling(i)
            final['num_epochs'].append(data['num_epochs'])
            final['learning rate'].append(data['learning rate'])
            final['loss'].append(data['loss'])
            final['accuracy_top-1'].append(data['accuracy_top-1'])
            final['accuracy_top-5'].append(data['accuracy_top-5'])
        else:
            data=JSON_datahandling(i)
            final={'num_epochs':[data['num_epochs']],'learning rate':[data['learning rate']],'loss':[data['loss']],'accuracy_top-1':[data['accuracy_top-1']],'accuracy_top-5':[data['accuracy_top-5']]}
        j=j+1
    return(pd.DataFrame.from_dict(final))


def print_table(urls):   #Writes the content of the previous jsons into a txt file (with some ASCII formatting) and a .csv file for handiling the data
    from prettytable import from_csv
    df=Result_table_gen(urls)
    with open('results/results.csv', 'w') as f:
        dfAsString = df.to_csv(header=True, index=False,sep=',')
        f.write(dfAsString)
        f.close()

    with open("results/results.csv") as f:
        table=from_csv(f)
        print(table)
        f.close

    with open("results/results.txt",'w') as f:
        table=table.get_string()
        f.write(table)
        f.close

