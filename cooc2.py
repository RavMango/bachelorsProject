import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import sys

 # Codes to count
codes_to_count = [
    "Assumption", "Component Behavior", "Constraints", "Decision Rule",
    "Design Configuration", "Quality Issue", "Requirements", 
    "Solution Benefits and Drawbacks", "Solution Comparison", "Solution Evaluation",
    "Solution Risks", "Solution Trade-off"
]

# Tags to count (make sure they're in alphabetical order)
tags = ["tag:existence", "tag:property", "tag:technology"]

# Tags to ignore (also alphabetical order)
ignore_tags = ["tag:notag", "tag:other", "tag:process",]

def counter(filename):
    # Load the .xlsx file into a dataframe
    df = pd.read_excel(filename)

    # Splitting the 'Reference' column
    df[['Start', 'End']] = df['Reference'].str.split('-', expand=True)

    # Convert 'Start' and 'End' columns to integer
    df['Start'] = pd.to_numeric(df['Start'].str.strip(), errors='coerce')
    df['End'] = pd.to_numeric(df['End'].str.strip(), errors='coerce')

    # Sorting
    df = df.sort_values(by=["Document", "Start", "End"])

    #remove any empty rows of 'Codes' or 'Quotation Content'
    df = df.dropna(subset=['Codes', 'Quotation Content'])

    # list that contains all possible combinations of 2 of the codes
    codes_combinations = list(itertools.combinations(codes_to_count, 2))
    # also combinations of same value
    for code in codes_to_count:
        codes_combinations.append((code, code))

    # set of all possible combinations of tags of all lengths
    tags_combinations = []
    for i in range(2, len(tags) + 1):
        tags_combinations += list(itertools.combinations(tags, i))

    # print(tags_combinations)

    #make a map to count the occurrences of each combination within a quotation
    codes_combinations_map_quotation = {}
    for combination in codes_combinations:
        codes_combinations_map_quotation[combination] = 0

    #make a map to count the occurrences of each combination within an email
    codes_combinations_map_email = {}
    for combination in codes_combinations:
        codes_combinations_map_email[combination] = 0

    tags_combinations_map = {}
    for combination in tags_combinations:
        tags_combinations_map[combination] = 0

    # list to hold the lenght of the 'Quotation Content' column for each AK concept
    quotation_length = {}
    for code in codes_to_count:
        quotation_length[code] = []

    #make a list to count the occurentes of each code and tag
    occurence_count = {}
    for code in codes_to_count:
        occurence_count[code] = 0
    for tag in tags:
        occurence_count[tag] = 0

    #make a list to count the ak concept occurences within a combo
    occurence_count_in_combos = {}
    for code in codes_to_count:
        occurence_count_in_combos[code] = 0

    tag_count = {}
    for tag in tags:
        for code in codes_to_count:
            tag_count[tag, code] = 0

    #list to contian all the ADD tags currently set for email
    in_tags = []

    # boolean to check if the email contains an AK concept
    contains_ak = False

    # list to hold all valid emails containing AK concepts
    valid_email_map = set()

    

    # initialize the current document variable
    current_doc = ""
    
    list_of_ak_concepts_in_email = []
    # ----------Iterate over the rows------------------------------
    for index, row in df.iterrows():
        # print (row['Document'], row['Reference'])

        # Get the word count
        word_count = len(row['Quotation Content'].split())
        # Get the codes
        codes = row['Codes'].split('\n')

        # if we've entered a new document, then we must reset the in_tags list
        if current_doc != row['Document']:
            current_doc = row['Document']
            if contains_ak:
                if len(in_tags) > 1:
                    for x in list(itertools.combinations(in_tags, len(in_tags))):
                        tags_combinations_map[x] += 1
                
                for code in in_tags:
                    occurence_count[code] += 1
                    # print(code, "++")
                contains_ak = False
            in_tags = []
            ignore = False
            # print("RESET")

        # for i in range(len(codes)):
        #     print(i, codes[i])

        for code in codes[:]:  # Creates a copy of the list for iteration
            # print(code)
            if code in ignore_tags and len(codes) == 1:
                ignore = True
                ak_flag = False
                codes.remove(code)
            elif code in tags:
                ignore = False
                ak_flag = False
            elif code not in codes_to_count:
                codes.remove(code)
            else:
                ak_flag = True
        
        # print(ignore, ak_flag, contains_ak)
        # print("in_tags: ", in_tags, "codes: ", codes)
                
        
        combos = list(itertools.combinations(codes, 2))

        #PRINT COMBOS
        #if there are combos (length of filtered Codes is greater than 1)
        # if combos:
        #     for combo in combos:
        #         print(combo)

            
        if code in tags:
            # print("TAGS: ", codes)
            if contains_ak:
                if len(in_tags) > 1:
                    for x in list(itertools.combinations(in_tags, len(in_tags))):
                        tags_combinations_map[x] += 1
                
                for code in in_tags:
                    occurence_count[code] += 1
                    # print(code, "++")
                contains_ak = False
            in_tags = codes
            ignore = False

            list_of_ak_concepts_in_email.sort()
            # print("Document: ", row['Document'])
            # print("Reference: ", row['Reference'])
            # print("list_of_ak_concepts_in_email: ", list_of_ak_concepts_in_email)
            combos2 = list(itertools.combinations(list_of_ak_concepts_in_email, 2))
            for combo in combos2:
                # print(combo)
                codes_combinations_map_email[combo] += 1
                occurence_count_in_combos[combo[0]] += 1
                occurence_count_in_combos[combo[1]] += 1
            list_of_ak_concepts_in_email = []

        # Check if the code is in the list
        elif ak_flag and not ignore:
            contains_ak = True
            # add document and reference to valid_email_map
            valid_email_map.add((row['Document'], row['Reference']))
            #iterate over codes and increment the count
            for combo in combos:
                codes_combinations_map_quotation[combo] += 1
                # print(combo, "++")
            for code in codes:
                if len(in_tags) > 1: #NORMALIZATION
                    occurence_count[code] += 1
                    list_of_ak_concepts_in_email.append(code)
                    for tag in in_tags:
                        # tag_count[tag, code] += 1/len(in_tags) # NORMALIZATION
                        tag_count[tag, code] += 1

                else:
                    occurence_count[code] += 1
                    list_of_ak_concepts_in_email.append(code)
                    for tag in in_tags:
                        tag_count[tag, code] += 1
                quotation_length[code].append(word_count)   
                # print(code, "++")
    print("codes_combinations_map")
    for key, value in codes_combinations_map_email.items():
        print(key, round(value))
    
    for code in codes_to_count:
        print(code, occurence_count_in_combos[code])

    # printing(codes_combinations_map_quotation, tags_combinations_map, occurence_count, quotation_length, valid_email_map, tags, tag_count)
    # printAKConceptsPerTag(tag_count, occurence_count)
    # RQ1figureMaker(tags, tag_count, occurence_count)
    RQ2figureMaker(quotation_length)

    # chi square test for the co-occurence of AK concepts within a quotation
    # chiSquare(codes_combinations_map, occurence_count, tags)

    # chi square test for the co-occurence of AK concepts within an email
    # chiSquare(codes_combinations_map_email, occurence_count_in_combos, tags)
        
        
def printing(codes_combinations_map, tags_combinations_map, occurence_count, quotation_length, valid_email_map, tags, tag_count):

    # Print the map
    for key, value in codes_combinations_map.items():
        print(key, round(value))
    print()

    for key, value in tags_combinations_map.items():
        print(key, round(value))
    print()
    
    total_count = 0
    for key, value in codes_combinations_map.items():
        if value > 0:
            total_count += value
    
    print("Total co-occurences: ", total_count)
    print()

    #print total occurence count
    for key, value in occurence_count.items():
        print(key, value)
    
    total_tag_count = 0
    #total tags occurence count
    for key, value in occurence_count.items():
        if key in tags:
            total_tag_count += value
    print()

    #print average quotation length of each AK concept
    for key, value in quotation_length.items():
        if value:
            print(key, round(sum(value)/len(value)))

    print()


    for tag in tags:
        for code in codes_to_count:
            print(tag, code, tag_count[tag, code])

    #Total Documents with AK concepts (every unique Document)
    print("Total Documents with AK concepts: ", len(set([x[0] for x in valid_email_map])))
        

    #Total emails with AK concepts
    print("Total emails with AK concepts: ", len(valid_email_map))

    print("Total tags count: ", total_tag_count)

    print("Total AK concepts count: ", round(sum(occurence_count.values()) - total_tag_count))

def printAKConceptsPerTag(tag_count, occurence_count):
    for tag in tags:
        print(tag, "(occured ", occurence_count[tag], " times)")
        for code in codes_to_count:
            print(code, round(tag_count[tag, code]))
        print()

def RQ1figureMaker(tags, tag_count, occurence_count):
    # pie chart of all the AK concepts
    total = 0
    for code in codes_to_count:
        total += occurence_count[code]
    
    labels = []
    sizes = []
    for code in codes_to_count:
        labels.append(code)
        sizes.append(occurence_count[code])

    # adds a \n to the label if it's too long
    for i in range(len(labels)):
        if len(labels[i]) > 22:
            labels[i] = labels[i][:22] + '\n' + labels[i][22:]


    plt.figure(figsize=(12, 10))
    wedges, texts, autotexts = plt.pie(sizes,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}% ({pct * total / 100:.0f})",
            startangle=140,
            wedgeprops=dict(width=0.3, edgecolor='w'),
            pctdistance=0.58
            )
    plt.axis('equal')
    plt.setp(texts + autotexts, size=12)
    plt.savefig("images2/ak_concepts_distribution_pie_chart.png", bbox_inches='tight')


    # pie charts for each ADD tag
    significant_codes = {}
    
    percentage_threshold = 4

    for tag in tags:
        for code in codes_to_count:
            significant_codes[code] = 0
        significant_codes["Other AK Concepts"] = 0

        for code in codes_to_count:
            total_ak = sum(tag_count[tag, code] for code in codes_to_count)
            if (tag_count[tag, code] / total_ak)*100 > percentage_threshold:
                significant_codes[code] = tag_count[tag, code]
            else:
                significant_codes.pop(code)
                significant_codes["Other AK Concepts"] += tag_count[tag, code]

        total_ak_in_tag = sum(significant_codes.values())

        # plot pie chart of the significant codes
        labels = []
        sizes = []
        for code in significant_codes:
            labels.append(code)
            sizes.append(significant_codes[code])

        # add a \n to the label if it's too long
        for i in range(len(labels)):
            if len(labels[i]) > 22:
                labels[i] = labels[i][:22] + '\n' + labels[i][22:]
        
        plt.figure(figsize=(10, 7))
        wedges, texts, autotexts = plt.pie(sizes,
                labels=labels,
                #print the percentage and the count of each code
                autopct=lambda pct: f"{pct:.1f}% ({pct * total_ak_in_tag / 100:.0f})",
                pctdistance=0.75,
                startangle=90)
        plt.tight_layout()
        plt.setp(texts + autotexts, size=12)
        plt.savefig(f"images2/distribution_{tag[4:]}.png")

def RQ2figureMaker(quotation_length):

    #box plot of the quotation length of each AK concept
    data = []
    labels = []
    for key, value in quotation_length.items():
        if value:
            data.append(value)
            labels.append(key)
    plt.figure(figsize=(10, 7))
    plt.boxplot(data,
                labels=labels,
                patch_artist=True,
                vert=False,
                showfliers=False,
                widths=0.7
                )
    plt.xlabel("Number of Words in Quotation", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("images2/word_count_distribution_per_ak_concept.png")

    #print mean, median and mode of the quotation length of each AK concept
    for code in codes_to_count:
        print(code, "Mean: ", round(np.mean(quotation_length[code])), 
              "Median: ", round(np.median(quotation_length[code])), 
              "Mode: ", round(np.median(quotation_length[code])), 
              "Spread: ", round(np.ptp(quotation_length[code])),
                "Variance: ", round(np.var(quotation_length[code])),
                "Standard Deviation: ", round(np.std(quotation_length[code]))
            )
        

def chiSquare(observed_map, occurence_count, tags):
    total_count = 0
    for key, value in occurence_count.items():
        if key not in tags:
            total_count += value
    print("Total count: ", total_count)

    #ecpected map
    expected_map = {}
    for key, value in observed_map.items():
        expected_map[key] = 0
    
    for key, value in expected_map.items():
        expected_map[key] = occurence_count[key[0]]*occurence_count[key[1]]/total_count
    
    #chi square test
    chi_square_map = {}
    p_value_map = {}
    for key, value in observed_map.items():
        # print (key[0], key[1], value)
        if value == 0:
            chi_square_map[key] = 1
            p_value_map[key] = 1
            continue
        current = value
        row_sum = occurence_count[key[0]]
        col_sum = occurence_count[key[1]]
        right = row_sum - current
        bottom = col_sum - current
        total_sum = total_count
        bottom_right = total_sum - row_sum - col_sum + current
        contingency_table = np.array(
            [
                [current, right],
                [bottom, bottom_right]
            ])
        # print(contingency_table)
        stat, p, dof, expected = chi2_contingency(contingency_table)
        chi_square_map[key] = stat
        p_value_map[key] = p
    # Adjusted alpha level for multiple comparisons
    adjusted_alpha = 0.05 / 78

    # Degrees of freedom
    df = 11*11

    # Find the critical value
    critical_value = chi2.ppf(1 - adjusted_alpha, df)
    print("Chi Square values (with p-values) greater than critical value:",
          critical_value)
    for key, value in chi_square_map.items():
        if value > critical_value:
            print(key, round(p_value_map[key], 2), round(value, 2))
         
if __name__ == "__main__":

    # if len(sys.argv) < 2:
    #     print("Usage: python cooc.py filename.xlsx")
    #     sys.exit(1)
    # filename = sys.argv[1]

    filename = 'Quotation Manager9.xlsx'
    counter(filename)