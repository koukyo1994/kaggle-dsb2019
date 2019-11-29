# kaggle-dsb2019

This is a repository for the Kaggle competition "Data Science Bowl 2019"

## Interpretation of event_code and some thoughts on them

Here's simple interpretations of each event_codes.
Note that I checked up the data to make this correspondence manually, not from any source of PBS KIDS App, so it may contain some mistakes. If you find some mistakes or find something odd in my interpretation, please give me a comment!

* `2000` : start of the session.

If you have a feature like *count of the event_code 2000 before that assessment*, then it means *count of the session before that assessment*. In my model this feature got comparatively high importance so I suppose this means those who did a lot of trial before are likely to get high performance at the assessment, which is quite intuitive.
One thing that I should note here is that I also got high importance on this feature in adversarial validation, which means it  display the distribution difference between the length of the history of train data and that of test data to some extent, so it is better to replace the feature with feature(s) which extract similar property, but do not display the distribution difference, if we have some good ideas on it.

* `2010`: exit from the session

In the train dataset, I found this event_code in the sessions of `Assessment`, `Game`, `Activity`, and in the test dataset, I found this in only the sessions of `Assessment`. This event_code appears to be the indication that the user go out from the session like deleting the tab or pushing the return button of the browser (this is just my guess). This code is always followed by a sample which has the code 2000, which means this code indicates the end of the session. But note that not all the sessions end with this code. On the contrary, sessions end with this event_code are not so many.

* `2020`:

* `3010`: Start of short animation.

Automatically repeated or mouse over.

* `3110`: End of short animation.

* `3020`: Try Again code.

This code only appears in `Game` and `Assessment`. If the player make a mistake, this code appears in the log, so it can be an indicator how many times the player made mistakes.

* `3120`:

* `3021`: (Usually) Success code.

This code also only appears in `Game` and `Assessment`. If the player make a correct choice, this code appears in the log. But in some case (`Bubble Bath` of `Game`) this code does not mean success but only indicate that

* `3121`:

* `4020`: Choice code.

* `4030`: Some action

* `4031`: End of action

* `4070`: Some action

Moving slider, Click?

## Event path

* Cauldron Filler (Assessment)
    1. Magma Peak - Level 1
    2. Sandcastle Builder (Activity)
    3. Slop Problem
    4. Scrub-A-Dub
    5. Watering Hole (Activity)
    6. Magma Peak - Level2
    7. Dino Drink
    8. Bubble Bath
    9. Bottle Filler (Activity)
    10. Dino Dive
    11. Cauldron Filler (Assessment)

* Tree Top City
    1. Tree Top City - Level 1
    2. Ordering Spheres
    3. All Star Sorting
    4. Costume Box
    5. Fireworks (Activity)
    6. 12 Monkeys
    7. Tree Top City - Level 2
    8. Flower Waterer (Activity)
    9. Pirate's Tale
    10. Mushroom Sorter (Assessment)
    11. Air Show
    12. Treasure Map
    13. Tree Top City - Level 3
    14. Crystals Rule
    15. Rulers
    16. Bug Measurer (Activity)
    17. Bird Measurer (Assessment)

* Crystal Caves
    1. Crystal Caves - Level 1
    2. Chow Time
    3. Balancing Act
    4. Chicken Balancer (Activity)
    5. Lifting Heavy Things
    6. Crystal Caves - Level 2
    7. Honey Cake
    8. Happy Camel
    9. Cart Balancer (Assessment)
    10. Leaf Leader
    11. Crystal Caves - Level 3
    12. Heavy, Heavier, Heaviest
    13. Pan Balance
    14. Egg Dropper (Activity)
    15. Chest Sorter (Assessment)
