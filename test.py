import json
max1,max2 = -10,-10
min1,min2 = 10,10
max3,max4 = -10,-10
min3,min4 = 10,10
with open("datas\\all_dir_sph_range_2.json") as f:
    datas = json.load(f)
for data in datas:
    if data["dir_sph"][0]>max1:
        max1 = data["dir_sph"][0]
    if data["dir_sph"][0]<min1:
        min1 = data["dir_sph"][0]
    if data["dir_sph"][1]>max2:
        max2 = data["dir_sph"][1]
    if data["dir_sph"][1]<min2:
        min2 = data["dir_sph"][1]
    if data["point_sph"][0]>max3:
        max3 = data["point_sph"][0]
    if data["point_sph"][0]<min3:
        min3 = data["point_sph"][0]
    if data["point_sph"][1]>max4:
        max4 = data["point_sph"][1]
    if data["point_sph"][1]<min4:
        min4 = data["point_sph"][1]
    
print(max1)
print(max2)
print(min1)
print(min2)