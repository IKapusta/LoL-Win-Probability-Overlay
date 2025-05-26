import os
import detection.area_detection
from detection.eocr import extract_numbers_with_easyocr
from detection.inference import classify_image
from detection.xp_bar import calculate_purple_percentage
from detection.dragon_detecting import count_dragon_icons
import detection.areas

def get_total_gold(screenshot_paths, output_folder = "./tmp"):
    """
    Extracts total gold for blue and red teams from the scoreboard.

    Args:
        screenshot_paths (list): List of paths to screenshots.
        output_folder (str): Folder to store upscaled areas.

    Returns:
        tuple: (blue_gold, red_gold)
    """    
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.gold, output_folder)
    
    def process_extracted_text(extracted_text):
            """
            Process the extracted text to ensure it is in the correct format (x.xK).

            Args:
                extracted_text (str): Text extracted from the screenshot.

            Returns:
                int: The processed gold amount as an integer.
            """
            if not extracted_text.endswith("K"):
                raise ValueError(f"Invalid format: {extracted_text} (expected 'x.xK')")
            
            if "." in extracted_text:
                return int(float(extracted_text[:-1]) * 1000)
            else:
                # Infer missing decimal point (e.g., "290K" -> "29.0K")
                inferred_value = f"{extracted_text[:-1][:-1]}.{extracted_text[:-1][-1]}"
                return int(float(inferred_value) * 1000)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_0.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path, allowList='0123456789.kK')
    blue_gold = process_extracted_text(extracted_text)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_1.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path, allowList='0123456789.kK')
    red_gold = process_extracted_text(extracted_text)

    
    return blue_gold, red_gold

def get_kills(screenshot_paths, output_folder = "./tmp"):
    """
    Extracts kills for both teams.

    Returns:
        tuple: (blue_kills, red_kills)
    """    
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.kills, output_folder)
    
    upscaled_path = os.path.join(output_folder, f"upscaled_area_0.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    blue_kills = int(extracted_text)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_1.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    red_kills = int(extracted_text)

    return blue_kills, red_kills

def get_cs(screenshot_paths, output_folder = "./tmp",time = 0., area = "late"):
    """
    Extracts total CS for both teams.

    Args:
        time (float): Game time in seconds.
        area (str): "late" or "early" to determine which areas to crop.

    Returns:
        tuple: (blue_total_cs, red_total_cs)
    """
    blue_total_cs = 10
    red_total_cs = 10

    def validate_and_correct_cs(cs_list, game_time):
        if game_time < 300 :
            return cs_list
        corrected_cs = cs_list.copy()

        max_expected_cs =  game_time // 10 + game_time / 30  

        for i in range(len(corrected_cs)):
            if corrected_cs[i] > max_expected_cs:
                print(f"Anomalous CS detected: {corrected_cs[i]}. (gametime is {game_time}, max expected cs is {max_expected_cs}) Correcting...")
                corrected_cs[i] = int(str(max_expected_cs)[0]+ str(cs_list[i])[1:])
        return corrected_cs

    if area == "late":
        areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.blue_cs, output_folder)
    else:
        areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.blue_cs_early, output_folder)

    rolewise_cs = []
    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        extracted_text = extract_numbers_with_easyocr(upscaled_path)
        rolewise_cs.append(int(extracted_text))
    
    rolewise_cs = validate_and_correct_cs(rolewise_cs, time)
    blue_total_cs = sum(rolewise_cs)

    
    if area == "late":
        areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.red_cs, output_folder)
    else:
        areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.red_cs_early, output_folder)

    rolewise_cs = []
    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        extracted_text = extract_numbers_with_easyocr(upscaled_path)
        rolewise_cs.append(int(extracted_text))
    
    rolewise_cs = validate_and_correct_cs(rolewise_cs,time)
    red_total_cs = sum(rolewise_cs)
    
    return blue_total_cs, red_total_cs

def get_turrets(screenshot_paths, output_folder = "./tmp"):
    """
    Extracts turret count for both teams.
    """    
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.turrets, output_folder)
    
    upscaled_path = os.path.join(output_folder, f"upscaled_area_0.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    blue_turrets = int(extracted_text)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_1.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    red_turrets = int(extracted_text)

    return blue_turrets, red_turrets

def get_grubs(screenshot_paths, output_folder = "./tmp"):
    """
    Extracts the number of grubs for both teams.
    """    
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.grubs, output_folder)
    
    upscaled_path = os.path.join(output_folder, f"upscaled_area_0.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    blue_grubs = int(extracted_text)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_1.png")
    extracted_text = extract_numbers_with_easyocr(upscaled_path)
    red_grubs = int(extracted_text)

    return blue_grubs, red_grubs


def get_xp(screenshot_paths, output_folder = "./tmp"):
    """
    Computes XP for both teams based on player levels and XP bar percentages.
    """

    def xp_from_level(levels):
        total_xp = 0
        for level in levels:
            base_xp = 280
            player_xp = 0
            for i in range(1,int(level)):
                player_xp += base_xp
                base_xp += 100
            player_xp += level%1 * base_xp
            total_xp += player_xp
        return total_xp
    
    blue_team = []
    red_team = []

    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.blue_levels, output_folder)
    
    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        level = classify_image(upscaled_path)
        print(level)
        if level == 0:
            raise ValueError     
        blue_team.append(int(level))
    
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.red_levels, output_folder)
    
    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        level = classify_image(upscaled_path)
        print(level)
        if level == 0:
            raise ValueError
        red_team.append(int(level))

    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.blue_xp, output_folder)
    
    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        percentage = calculate_purple_percentage(upscaled_path)
        blue_team[i] += percentage/100

    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.red_xp, output_folder)

    for i in range(len(areas_of_interest)):
        upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
        percentage = calculate_purple_percentage(upscaled_path)
        red_team[i] += percentage/100
    
    blue_xp = xp_from_level(blue_team)
    red_xp = xp_from_level(red_team)
    
    return blue_xp, red_xp
   
def get_gametime(screenshot_paths, output_folder = "./tmp"):
    """
    Extracts game time and calculates a time normalization factor.
    """  
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.gametime, output_folder)

    upscaled_path = os.path.join(output_folder, f"upscaled_area_0.png")
    time = extract_numbers_with_easyocr(upscaled_path)
    time = float(time)
    if time > 9999 : 
        time /= 10
   
    timefactor = 1000. / time 
    
    return time, timefactor

def get_dragons(screenshot_paths, output_folder = "./tmp", upscaling = True):
    """
    Counts dragon icons for both teams.
    """
    areas_of_interest = detection.area_detection.process_screenshots(screenshot_paths, detection.areas.dragons, output_folder, upscaling=upscaling)
    
    blue_img = os.path.join(output_folder, f"upscaled_area_0.png")
    red_img = os.path.join(output_folder, f"upscaled_area_1.png")
    
    blue_dragons = count_dragon_icons(blue_img)
    red_dragons = count_dragon_icons(red_img)
    
    return blue_dragons, red_dragons