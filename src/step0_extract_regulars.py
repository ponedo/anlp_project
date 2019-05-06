# District Court Ruling
# Circuit Court Ruling
# District Court
# Plaintiff
# Defendant
# Appellant
# Appellee

import re
import os
from utils import read_raw_file, raw_file_list, regular_results_dir
from preprocess import truncate, compiled_regex_needed


verdict_dict = [
    "affirmed", 
    "reversed", 
    "vacated", 
    "remanded",  
]


def extract_regular_stuffs(i):
    raw_text = read_raw_file(i)
    
    # truncate pieces first
    raw_text = truncate(raw_text).strip("\n").strip()
    
    # split into lines
    # iterate over lines to find case number, parties and date
    splited_text = raw_text.split("\n")
    case_num = splited_text[0]

    parties_found, date_found, verdict_found = False, False, False
    pl_apla, pl_aple, df_apla, df_aple = [], [], [], []
    pl, df, apla, aple = [], [], [], []
    verdicts = []

    for idx, text in enumerate(splited_text):
    
        # involved sides appear around "v."
        if not parties_found and text == "v.":
            parties_found = True

            # find the parties above "v."
            j = 1
            while True:
                party = splited_text[idx - j]
                partitioned_party = party.partition(", ")
                name = "".join(partitioned_party[:-2])
                role =  partitioned_party[-1].lower()

                # find if they are plaintiff/defendant/appellant/appellee
                if "plaintiff" in role:
                    pl.append(name)
                    if "appellant" in role:
                        apla.append(name)
                        pl_apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                        pl_aple.append(name)
                elif "defendant" in role:
                    df.append(name)
                    if "appellant" in role:
                        apla.append(name)
                        df_apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                        df_aple.append(name)
                else:
                    if "appellant" in role:
                        apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                    else:
                        raise Exception("A party has no role")
                
                j += 1
                if splited_text[idx - j] != "and":
                    break
                j += 1
            
            # find the parties beneath "v."
            j = 1
            while True:
                party = splited_text[idx + j]
                partitioned_party = party.partition(", ")
                name = "".join(partitioned_party[:-2])
                role =  partitioned_party[-1].lower()

                # find if they are plaintiff/defendant/appellant/appellee
                if "plaintiff" in role:
                    pl.append(name)
                    if "appellant" in role:
                        apla.append(name)
                        pl_apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                        pl_aple.append(name)
                elif "defendant" in role:
                    df.append(name)
                    if "appellant" in role:
                        apla.append(name)
                        df_apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                        df_aple.append(name)
                else:
                    if "appellant" in role:
                        apla.append(name)
                    elif "appellee" in role:
                        aple.append(name)
                    else:
                        raise Exception("A party has no role")
                
                j += 1
                if splited_text[idx + j] != "and":
                    break
                j += 1

        # find date
        if not date_found and text == "|":
        # elif not date_found and text == "|":
            date_found == True
            try:
                date = re.findall(compiled_regex_needed, splited_text[idx + 1])[0]
            except IndexError:
                date_found == False

        if not verdict_found:
            splitted_text = text.strip(".").lower().split(" and ")
            for possible_verdict in splitted_text:
                if possible_verdict not in verdict_dict:
                    break
                verdicts.append(possible_verdict)
            else:
                verdict_found = True
            
    # infer district verdict
    if pl_apla and df_aple:
        district_ruling = "defendant(s) won"
    elif pl_aple and df_apla:
        district_ruling = "plaintiff(s) won"
    else:
        district_ruling = "?"

    # find district court
    # the first appearance of string "the united states district court" is where it is
    raw_text = raw_text.lower()
    distrcit_court_pos = raw_text.find("the united states district court")
    i = distrcit_court_pos
    while True:
        if raw_text[i] == ",":
            district_court = raw_text[distrcit_court_pos:i]
            break
        i += 1

    return {
        "case_num": case_num, 
        "date": date, 
        "plaintiff": pl, 
        "defendant": df, 
        "appellant": pl_apla + df_apla, 
        "appellee": pl_aple + df_aple, 
        "district_court": district_court, 
        "district_ruling": district_ruling, 
        "circuit_ruling": verdicts
    }


if __name__ == "__main__":
    for i in range(len(raw_file_list)):
        filename = raw_file_list[i]
        regular_result = extract_regular_stuffs(i)
        with open(os.path.join(regular_results_dir, filename), "w", encoding="cp950") as f:
            for k, v in regular_result.items():
                f.write(k + ":")
                if isinstance(v, list):
                    for e in v:
                        f.write("\t" + str(e))
                else:
                    f.write("\t" + str(v))
                f.write("\n\n")