repo=[]
id=1
with open('guacamol_250 copy.log','r') as f:
    for line in f:
        if 'Done' in line and 'Tanimoto' in line:
            dct={}
            indx = 46
            new_str = line[indx:-1]
            splits = new_str.split('@')
            name = splits[0]
            dct["NAME"] = name
            dct["SMILES"] = id
            data = splits[1].split(',')
            node_data = data[0].split('=')
            dct["NODES"]=int(node_data[1])

            solution_time_data = data[1].split('=')
            dct["SOLUTION_TIME"] = float(solution_time_data[1])
            dct["NUM_ROUTES"] = 0
            dct["FINAL_NUM_RXN_MODEL_CALLS"] = min(float(solution_time_data[1]),500)
            dct["final_num_value_model_calls"] = 1000
            repo.append(dct)
            # print(node_data)
            id+=1
print(repo)

# SMILES,NAME,NODES,SOLUTION_TIME,NUM_ROUTES,FINAL_NUM_RXN_MODEL_CALLS,final_num_value_model_calls
f = open('results_no_tan.csv','a')

for entry in repo:
    f.write(f'{entry["SMILES"]},{entry["NAME"]},{entry["NODES"]},{entry["SOLUTION_TIME"]},{entry["NUM_ROUTES"]},{entry["FINAL_NUM_RXN_MODEL_CALLS"]},{entry["final_num_value_model_calls"]}\n')
f.close()
print("WROTE TO CSV")