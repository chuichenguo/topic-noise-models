text = """0	1.66667	Tesla battery LA Supercharger pack years technology dealers swap job stop half network faster NY future vid feature largest SolarCity 
1	1.66667	high due Earth Launch super weather fast Great Love day expected station Landing meeting impact size opening Friday personally solution 
2	1.66667	fire months vehicle mins tunnel Boring air launches day set return Full Company JohnGardi thing reusable earlier Hawthorne Test progress 
3	1.66667	Model 3 version series dual cameras Future tests Review smaller lives saved SUV Congratulations radar customer EddyJahn gen Production digging 
4	1.66667	2 today 1 awesome unveil early road engines China orders fine morning ElectrekCo point Climate order factory 3 Liftoff Europe 
5	1.66667	Mars AI change OpenAI climate worlds Elon wont risk hours Worth Musk mashable complex newscientist favor reading matters gdoehne Trump 
6	1.66667	Falcon 9 launch SpaceX Cape Canaveral Heavy vertical window pad opens Sunday Air UTC Pad doors LZ1 Force 39A satellites 
7	1.66667	TeslaMotors RT http govt plane VoltzCoreAudio operation customers ride evidence installation planet splashdown lukealization irony good water safety gasoline Teslas 
8	1.66667	SpaceX Dragon launch NASA RT SpaceStation attempt spacecraft cargo advance Pacific footage Crew Falcon9 astronauts targeted date Ludicrous 3D apply 
9	1.66667	people great Yeah lot dont love announcement make watch short find waitbutwhy taking beautiful dead science race True call visit 
10	1.66667	car Tesla driving owners amazing put happy highest business shows pay Consumer full Reports mass tap phone left Id days 
11	1.66667	year miles Tesla IDAACarmack full 10 trip big show public idea dont world truck upgrade rate Grasshopper thing fleet misleading 
12	1.66667	SpaceX orbit mission Space space point target satellite Station deployed moon satellites advanced km Mach red transfer orbital ms altitude 
13	1.66667	RT elonmusk httptco person Texas details view 15 teslamotors issue elonmusks bigger Happy build Australia Doesnt customer isnt TheOnion 50 
14	1.66667	company service side important auto Ive drive rear making give care Superchargers ground California tech heavy EV longer read clear 
15	1.66667	RT Im future made WIRED TeslaMotors human home flying free testing baby start transport talk Yup missions suggestions accident Slate 
16	1.66667	Good article FredericLambert tax life max carbon start make heard cool country time Dont humanity camera isnt costs move turned 
17	1.66667	stage video upper systems burn low worth oxygen Read pressure tank mcannonbrookes Live autonomous landed flight based boost helium record 
18	1.66667	Autopilot week end RT Teslas release HW2 support days post ready built confirmed Hoping mode line material crash 80 correct 
19	1.66667	true 4 level data small story gas charge companies piece httpt back media easy reach case minutes launch News 2014 
20	1.66667	rocket flight back 1st thrust center engine wrong product hit part SmileSimplify cover close F9 underway perfect shot screen successful 
21	1.66667	landing hard Rocket rocket land droneship booster ship velocity design owner chance reentry landings drone static hover hardware deploy legs 
22	1.66667	good time working didnt works def bonnienorman money experience world action ago 7 System long vehicles add Meet TX Florida 
23	1.66667	cars production work electric control years sales crazy cost kids price engineering means selfdriving travel false 50 good Model thinking 
24	1.66667	solar energy power system roof long agree grid glass batteries Solar fins SolarCity electricity times interior biggest appreciation America AMA 
25	1.66667	Tesla live tomorrow Watch things 20 direct machine California webcast play answer wasnt avoid 90 force fixed Wow favorite autopilot 
26	1.66667	coming software needed speed update weeks month weekend vicentes TeslaRoadTrip MacTechGenius radar complete Youre hold limited buy motor improvements feels 
27	1.66667	60 range mph performance sec 14 100 mile tmrw verge 30 0 Gigafactory Coming Roadster P100D improve P85D kWh pretty 
28	1.66667	5 real 6 makes doesnt plan make bad sense spaceship safety tested businessinsider study ft Recode hate drive photos height 
29	1.66667	test Hyperloop team open Btw competition Cool top vacuum appreciated Congrats fuel bring danahull low fairing tomorrows building pods pod"""

import pandas as pd
from collections import Counter
import math

# 讀取CSV文件
tweets_df = pd.read_csv("data/data_elonmusk_preprocess5.csv")
#讀取文檔
documents = tweets_df.iloc[:, 0].tolist()

# 單詞分割
def tokenize(text):
    return text.lower().split()

# 創建單詞頻率計數器
word_counts = Counter()
for doc in documents:
    word_counts.update(tokenize(doc))

# 總文檔數
num_docs = len(documents)

# 函数计算单词共现概率
def word_cooccurrence_probability(word1, word2):
    count = 0
    for doc in documents:
        tokens = tokenize(doc)
        if word1 in tokens and word2 in tokens:
            count += 1
    return count / num_docs

# 函数计算NPMI
def npmi(word1, word2):
    p_word1 = word_counts[word1] / sum(word_counts.values())
    p_word2 = word_counts[word2] / sum(word_counts.values())
    p_word1_word2 = word_cooccurrence_probability(word1, word2)
    if p_word1_word2 == 0:
        return -1 # 避免除以零
    else:
        return (math.log(p_word1_word2 / (p_word1 * p_word2)) / -math.log(p_word1_word2))
lines = text.split('\n')

topics = [line.split()[2:] for line in lines]

avg=0

for idx, topic in enumerate(topics):
    total_npmi = 0
    pair_count = 0

    for i in range(len(topic)):
        for j in range(i + 1, len(topic)):
            npmivalue = npmi(topic[i], topic[j])
            if npmivalue != -1:
                total_npmi += npmivalue
                
    pair_count = len(topic)*(len(topic)-1)/2
    # Calculate the average NPMI for the current topic
    if pair_count > 0:
        average_npmi = total_npmi / pair_count
        avg=avg+average_npmi
        print(f"Topic {idx + 1}: Average NPMI = {average_npmi:.4f}")
    else:
        print(f"Topic {idx + 1}: No valid pairs for NPMI calculation")
print(avg/len(topics))
