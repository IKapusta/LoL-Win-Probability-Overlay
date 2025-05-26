import core.sample_builder_utils
import pandas as pd



def sample_to_dataframe(sample, t=10):
    """
    Converts the constructed sample dictionary into a DataFrame for model prediction.
    Missing features are filled with zeros if not provided.

    Args:
        sample (dict): Dictionary of extracted features from screenshots.
        minute (int): The closest time frame minute (10, 15, 20, or 25).

    Returns:
        pd.DataFrame: Single-row DataFrame in the expected format for prediction.
    """

    feature_cols = [
        'dragons', 'opp_dragons', 
        'void_grubs', 'opp_void_grubs',
        'towers', 'opp_towers',
        f'goldat{t}', f'xpat{t}', f'csat{t}', f'opp_goldat{t}', f'opp_xpat{t}', f'opp_csat{t}',
        f'golddiffat{t}', f'xpdiffat{t}', f'csdiffat{t}', f'killsat{t}',
        f'opp_killsat{t}'
    ]

    filled_sample = {col: sample.get(col, 0) for col in feature_cols}
    df = pd.DataFrame([filled_sample])

    return df



def construct_sample(screenshot_paths, output_folder="./tmp"):
    sample = {}
    print("Starting to construct sample...")

    gametime, timefactor = core.sample_builder_utils.get_gametime(screenshot_paths, output_folder)
    print(f"Game time: {gametime}, Time factor: {timefactor}")

    # Determine closest time milestone
    if gametime < 1230:
        minute = 10
    elif gametime < 1730:
        minute = 15
    elif gametime < 2230:
        minute = 20
    else:
        minute = 25
    print(f"Building sample for minute {minute}")

    # CS
    print("Extracting cs...")
    cs_area = "early" if gametime < 900 else "late"
    blue_cs, red_cs = core.sample_builder_utils.get_cs(screenshot_paths, output_folder, gametime, area=cs_area)
    sample[f'csat{minute}'] = blue_cs
    sample[f'opp_csat{minute}'] = red_cs
    sample[f'csdiffat{minute}'] = blue_cs - red_cs

    # kills
    print("Extracting kills...")
    blue_kills, red_kills = core.sample_builder_utils.get_kills(screenshot_paths, output_folder)
    sample[f'killsat{minute}'] = blue_kills
    sample[f'opp_killsat{minute}'] = red_kills

    # gold
    print("Extracting gold...")
    blue_gold, red_gold = core.sample_builder_utils.get_total_gold(screenshot_paths, output_folder)
    sample[f'goldat{minute}'] = blue_gold
    sample[f'opp_goldat{minute}'] = red_gold
    sample[f'golddiffat{minute}'] = blue_gold - red_gold

    # XP 
    print("Extracting xp...")
    blue_xp, red_xp = core.sample_builder_utils.get_xp(screenshot_paths, output_folder)
    sample[f'xpat{minute}'] = blue_xp
    sample[f'opp_xpat{minute}'] = red_xp
    sample[f'xpdiffat{minute}'] = blue_xp - red_xp

    # Dragons 
    print("Extracting dragons...")
    blue_dragons, red_dragons = core.sample_builder_utils.get_dragons(screenshot_paths, output_folder)
    sample['dragons'] = blue_dragons
    sample['opp_dragons'] = red_dragons

    # Towers
    print("Extracting towers...")
    blue_towers, red_towers = core.sample_builder_utils.get_turrets(screenshot_paths, output_folder)
    sample['towers'] = blue_towers
    sample['opp_towers'] = red_towers

    # Void Grubs
    print("Extracting grubs...")
    blue_grubs, red_grubs = core.sample_builder_utils.get_grubs(screenshot_paths,output_folder)
    sample['void_grubs'] = blue_grubs
    sample['opp_void_grubs'] = red_grubs
    
    
    print(sample)

    sample = sample_to_dataframe(sample,minute)

    print("Sample construction complete!")
    return sample, minute, gametime







if __name__ == "__main__":
    screenshot_paths = ["./screenshots/output.png"]
    sample = construct_sample(screenshot_paths)
    #print(sample)
