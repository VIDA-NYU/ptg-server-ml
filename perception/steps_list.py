import numpy as np

# step_ids = np.array([
#     'start',    
#     'end',    
#     "Place tortilla on cutting board.",     
#     "Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla",# leaving 1/2-inch", #uncovered at the edges.",     
#     "Clean the knife by wiping with a paper towel.",     
#     "Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",     
#     "Clean the knife by wiping with a paper towel.",     
#     "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick.",# Roll it tight",# enough to prevent gaps, but not so tight that the filling leaks.",     
#     "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",     
#     "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin",# between the last",# toothpick and the end of the roll. Discard ends.",     
#     "Slide floss under the tortilla, perpendicular to the length of the roll.",# Place the floss halfway", #between two toothpicks.",     
#     "Cross the two ends of the floss over the top of the tortilla roll. Holding",# one end of the floss in", #each hand, pull the floss ends in opposite directions to slice.",     
#     "Continue slicing with floss to create 5 pinwheels.",     
#     "Place the pinwheels on a plate.",    
#     "Measure 12 ounces of cold water and transfer to a kettle.",     
#     "Assemble the filter cone.  Place the dripper on top of a coffee mug.",     
#     "Prepare the filter insert by folding the paper filter in half to create a semi-circle",# and in half", #again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone.",     
#     "Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand",#, about 20 seconds. Transfer the grounds to the filter cone.",     
#     "Check the temperature of the water.",     
#     "Pour a small amount of water in the filter to wet the grounds. Wait about 30 seconds.",     
#     "Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill", #beyond the top of the paper filter.",     
#     "Let the coffee drain completely into the mug before removing the dripper. Discard the paper",# filter and coffee grounds.",    
#     "Place the paper cupcake liner inside the mug. Set aside.",     
#     "Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.",     
#     "Whisk to combine.",     
#     "Measure and add the oil, water, and vanilla to the bowl.",     
#     "Whisk batter until no lumps remain.",     
#     "Pour batter into prepared mug.",     
#     "Microwave the mug and batter on high power for 60 seconds.",     
#     "Check if the cake is done by inserting and toothpick into the center of the cake and then",# removing. If wet batter clings to the toothpick, microwave for an additional 5 seconds. If the toothpick comes out clean, continue.",     
#     "Invert the mug to release the cake onto a plate. Allow to cool until it is no longer hot",# to the touch, then carefully remove paper liner.",     
#     "While the cake is cooling, prepare to pipe the frosting. Scoop 4 spoonfuls of chocolate", #frosting into a zip-top bag and seal, removing as much air as possible.",     
#     "Use scissors to cut one corner from the bag to create a small opening 1/4-inch in diameter.",     
#     "Squeeze the frosting through the opening to apply small dollops of frosting to the plate",# in a circle around the base of the cake."
# ])
step_named_ids = np.array([
    "|OTHER",   # ADDED
    '|start',    
    '|end', 

    # pinwheels
    "1|place tortilla",
    "2|scoop and spread nut butter",
    "3|clean knife",
    "4|scoop and spread jelly",
    "5|clean knife",
    "6|roll tortilla",
    "7|insert toothpicks",
    "8|trim tortilla ends",
    "9|slide floss underneath tortilla",
    "10|slice tortilla with floss",
    "11|slice tortilla into 5 pinwheels",
    "12|place pinwheels on plate",
    # "Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla",# leaving 1/2-inch", #uncovered at the edges.",     
    # "Clean the knife by wiping with a paper towel.",     
    # "Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",     
    # "Clean the knife by wiping with a paper towel.",     
    # "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick.",# Roll it tight",# enough to prevent gaps, but not so tight that the filling leaks.",     
    # "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",     
    # "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin",# between the last",# toothpick and the end of the roll. Discard ends.",     
    # "Slide floss under the tortilla, perpendicular to the length of the roll.",# Place the floss halfway", #between two toothpicks.",     
    # "Cross the two ends of the floss over the top of the tortilla roll. Holding",# one end of the floss in", #each hand, pull the floss ends in opposite directions to slice.",     
    # "Continue slicing with floss to create 5 pinwheels.",     
    # "Place the pinwheels on a plate.",  

    # coffee 
    "1|measure water",
    "2|place coffee dripper on mug", 
    "3|fold filter into quarters", 
    "4|weigh coffee beans and grind",
    "5|measure water temperature",
    "6|pour water onto grounds",
    "7|slowly pour rest of water",
    "8|let water drain then remove dripper",
    # "Measure 12 ounces of cold water and transfer to a kettle.",     
    # "Assemble the filter cone.  Place the dripper on top of a coffee mug.",     
    # "Prepare the filter insert by folding the paper filter in half to create a semi-circle",# and in half", #again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone.",     
    # "Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand",#, about 20 seconds. Transfer the grounds to the filter cone.",     
    # "Check the temperature of the water.",     
    # "Pour a small amount of water in the filter to wet the grounds. Wait about 30 seconds.",     
    # "Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill", #beyond the top of the paper filter.",     
    # "Let the coffee drain completely into the mug before removing the dripper. Discard the paper",# filter and coffee grounds.",    

    "1|place cupcake liner in mug",
    "2|measure dry ingredients",
    "3|whisk to combine", 
    "4|add oil, water, vanilla", 
    "5|whisk batter until smooth", 
    "6|pour batter into mug",
    "7|microwave mug for 60s", 
    "8|check cake with toothpick", 
    "9|invert mug to release cake", 
    "10|prepare frosting bag", 
    "11|cut frosting bag corner", 
    "12|apply frosting to plate", 
    # "Place the paper cupcake liner inside the mug. Set aside.",    
    # "Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.",     
    # "Whisk to combine.",     
    # "Measure and add the oil, water, and vanilla to the bowl.",     
    # "Whisk batter until no lumps remain.",     
    # "Pour batter into prepared mug.",     
    # "Microwave the mug and batter on high power for 60 seconds.",     
    # "Check if the cake is done by inserting and toothpick into the center of the cake and then",# removing. If wet batter clings to the toothpick, microwave for an additional 5 seconds. If the toothpick comes out clean, continue.",     
    # "Invert the mug to release the cake onto a plate. Allow to cool until it is no longer hot",# to the touch, then carefully remove paper liner.",     
    # "While the cake is cooling, prepare to pipe the frosting. Scoop 4 spoonfuls of chocolate", #frosting into a zip-top bag and seal, removing as much air as possible.",     
    # "Use scissors to cut one corner from the bag to create a small opening 1/4-inch in diameter.",     
    # "Squeeze the frosting through the opening to apply small dollops of frosting to the plate",# in a circle around the base of the cake."
    # tourniquet
    "1|place tourniquet above wound",
    "2|pull tourniquet tight",
    "3|apply strap",
    "4|turn windlass until tight",
    "5|lock windlass into keeper",
    "6|pull strap over windlass keeper",
    "7|secure strap and windlass",
    "8|mark time on strap with marker",
    # 34:"Place tourniquet over affected extremity 2-3 inches above wound site. ",
    # 35:"Pull tourniquet tight.",
    # 36:"Apply strap to strap body.",
    # 37:"Turn windless clock wise or counter clockwise until hemorrhage is controlled.",
    # 38:"Lock windless into the windless keeper.",
    # 39:"Pull remaining strap over the windless keeper.",
    # 40:"Secure strap and windless keeper with keeper securing device.",
    # 41:"Mark time on securing device strap with permanent marker.",
])

step_ids = np.array([
    "-3",#'OTHER',
    "-2",#'|start',    
    "-1",#'|end',  

    # pinwheels
    "0",#"1|place tortilla",
    "1",#"2|scoop and spread nut butter",
    "2",#"3|clean knife",
    "3",#"4|scoop and spread jelly",
    "4",#"5|clean knife",
    "5",#"6|roll tortilla",
    "6",#"7|insert toothpicks",
    "7",#"8|trim tortilla ends",
    "8",#"9|slide floss underneath tortilla",
    "9",#"10|slice tortilla with floss",
    "10",#"11|slice tortilla into 5 pinwheels",
    "11",#"12|place pinwheels on plate",

    # coffee 
    "0",#"1|measure water",
    "1",#"2|place coffee dripper on mug", 
    "2",#"3|fold filter into quarters", 
    "3",#"4|weigh coffee beans and grind",
    "4",#"5|measure water temperature",
    "5",#"6|pour water onto grounds",
    "6",#"7|slowly pour rest of water",
    "7",#"8|let water drain then remove dripper",
    
    # mugcake
    "0",#"1|place cupcake liner in mug",
    "1",#"2|measure dry ingredients",
    "2",#"3|whisk to combine", 
    "3",#"4|add oil, water, vanilla", 
    "4",#"5|whisk batter until smooth", 
    "5",#"6|pour batter into mug",
    "6",#"7|microwave mug for 60s", 
    "7",#"8|check cake with toothpick", 
    "8",#"9|invert mug to release cake", 
    "9",#"10|prepare frosting bag", 
    "10",#"11|cut frosting bag corner", 
    "11",#"12|apply frosting to plate", 

    # tourniquet
    "0",#"1|place tourniquet above wound",
    "1",#"2|pull tourniquet tight",
    "2",#"3|apply strap",
    "3",#"4|turn windlass until tight",
    "4",#"5|lock windlass into keeper",
    "5",#"6|pull strap over windlass keeper",
    "6",#"7|secure strap and windlass",
    "7",#"8|mark time on strap with marker",
])


common_step_mask = np.array([0, 1, 2])
recipe_step_mask = {
    'pinwheels':  np.concatenate([1 + np.arange(2, 14), np.array([0])]).astype(int),
    'coffee':     np.concatenate([1 + np.arange(14, 22), np.array([0])]).astype(int),
    'mugcake':    np.concatenate([1 + np.arange(22, 34), np.array([0])]).astype(int),
    'tourniquet': np.concatenate([1 + np.arange(34, 42), np.array([0])]).astype(int),
}


RECIPES = {
    'pinwheels':  [
        "1|place tortilla",
        "2|scoop and spread nut butter",
        "3|clean knife",
        "4|scoop and spread jelly",
        "5|clean knife",
        "6|roll tortilla",
        "7|insert toothpicks",
        "8|trim tortilla ends",
        "9|slide floss underneath tortilla",
        "10|slice tortilla with floss",
        "11|slice tortilla into 5 pinwheels",
        "12|place pinwheels on plate",
    ],
    'coffee':     [
        "1|measure water",
        "2|place coffee dripper on mug", 
        "3|fold filter into quarters", 
        "4|weigh coffee beans and grind",
        "5|measure water temperature",
        "6|pour water onto grounds",
        "7|slowly pour rest of water",
        "8|let water drain then remove dripper",
    ],
    'mugcake':    [
        "1|place cupcake liner in mug",
        "2|measure dry ingredients",
        "3|whisk to combine", 
        "4|add oil, water, vanilla", 
        "5|whisk batter until smooth", 
        "6|pour batter into mug",
        "7|microwave mug for 60s", 
        "8|check cake with toothpick", 
        "9|invert mug to release cake", 
        "10|prepare frosting bag", 
        "11|cut frosting bag corner", 
        "12|apply frosting to plate", 
    ],
    'tourniquet': [
        "1|place tourniquet above wound",
        "2|pull tourniquet tight",
        "3|apply strap",
        "4|turn windlass until tight",
        "5|lock windlass into keeper",
        "6|pull strap over windlass keeper",
        "7|secure strap and windlass",
        "8|mark time on strap with marker",
    ],
}

common_labels = np.array(["|OTHER"])

RECIPE_STEP_LABELS = {
    k: np.concatenate([np.array(v), common_labels])
    for k, v in RECIPES.items()
}

RECIPE_STEP_IDS = {
    k: np.concatenate([np.arange(len(v)), -1 - np.arange(len(common_labels))])
    for k, v in RECIPES.items()
}

# recipe_output_mask = {}
# for k in ['tourniquet', ]
