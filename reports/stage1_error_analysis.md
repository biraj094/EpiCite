# Stage1 Error Analysis

## Claim
- sentence: Pierrot is renowned for their high end televisions.
  true/pred: Claim -> Background
  conf_pred: 0.5520, conf_true: 0.2592
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Temple Grandin stars Melissa McCarthy as Temple Grandin.
  true/pred: Claim -> Fact
  conf_pred: 0.8885, conf_true: 0.0621
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Chinese people can be associated with Ireland through ethnicity.
  true/pred: Claim -> Fact
  conf_pred: 0.8330, conf_true: 0.0817
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The Fly is a science fiction character.
  true/pred: Claim -> Fact
  conf_pred: 0.8418, conf_true: 0.1178
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Titanic's crew members all died on the ship.
  true/pred: Claim -> Evidence
  conf_pred: 0.7954, conf_true: 0.0954
  reason: Claim phrasing resembles supporting statement rather than assertion.
- sentence: Steffi Graf won 6 consecutive majors.
  true/pred: Claim -> Fact
  conf_pred: 0.5636, conf_true: 0.2099
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Anne Hathaway had a cameo in Rachel Getting Married.
  true/pred: Claim -> Fact
  conf_pred: 0.9314, conf_true: 0.0156
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The Bold and the Beautiful has an estimated 26.2 billion viewers.
  true/pred: Claim -> Fact
  conf_pred: 0.6210, conf_true: 0.0834
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Telangana has a trade agreement with the 12th largest state in India.
  true/pred: Claim -> Fact
  conf_pred: 0.8575, conf_true: 0.0598
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: A team from South Carolina is the Carolina Panthers.
  true/pred: Claim -> Fact
  conf_pred: 0.9057, conf_true: 0.0440
  reason: Boundary between epistemic categories appears ambiguous.

## Fact
- sentence: The Citadelle Laferrière is seen in Nord, Haiti.
  true/pred: Fact -> Claim
  conf_pred: 0.3784, conf_true: 0.2430
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: How to Be is a independent film released in 2008.
  true/pred: Fact -> Claim
  conf_pred: 0.5789, conf_true: 0.3451
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The Divergent Series' first movie was directed by Neil Burger in 2014.
  true/pred: Fact -> Claim
  conf_pred: 0.8354, conf_true: 0.1270
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: NBC worked with The Carmichael Show.
  true/pred: Fact -> Background
  conf_pred: 0.4872, conf_true: 0.3697
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Minor League Baseball compete at levels below a baseball league.
  true/pred: Fact -> Claim
  conf_pred: 0.4593, conf_true: 0.4279
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Martin Scorsese founded the World Cinema Foundation in 2007.
  true/pred: Fact -> Claim
  conf_pred: 0.6599, conf_true: 0.2507
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Portugues is a Western Romance Language.
  true/pred: Fact -> Claim
  conf_pred: 0.4859, conf_true: 0.4555
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Ad buyers use jingles in commercials.
  true/pred: Fact -> Background
  conf_pred: 0.5855, conf_true: 0.3947
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Brigitte Macron's husband is a president.
  true/pred: Fact -> Background
  conf_pred: 0.5287, conf_true: 0.1491
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Joan Cusack worked with Mike Nichols.
  true/pred: Fact -> Background
  conf_pred: 0.5100, conf_true: 0.4511
  reason: Boundary between epistemic categories appears ambiguous.

## Evidence
- sentence: There is no chance to renovate to save a history site once it's gone
  true/pred: Evidence -> Claim
  conf_pred: 0.9079, conf_true: 0.0875
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: It is unlikely to buy things impulsively for adults who have concern of arranging disposable income
  true/pred: Evidence -> Claim
  conf_pred: 0.9000, conf_true: 0.0960
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: the environment today is serious damaged
  true/pred: Evidence -> Claim
  conf_pred: 0.8782, conf_true: 0.1162
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: as a matter of fact, using technology or advanced facilities do not make food lose its nutrition and quality
  true/pred: Evidence -> Claim
  conf_pred: 0.5478, conf_true: 0.4424
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: children can learn social skills when they have a job
  true/pred: Evidence -> Claim
  conf_pred: 0.6545, conf_true: 0.3405
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: By relating well to people, we are likely to have a lot of friends
  true/pred: Evidence -> Claim
  conf_pred: 0.8719, conf_true: 0.1235
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Computers expose children to vulgar materials
  true/pred: Evidence -> Claim
  conf_pred: 0.9273, conf_true: 0.0639
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The sixth season premiered on September 20 , 2016 .
  true/pred: Evidence -> Fact
  conf_pred: 0.6713, conf_true: 0.0664
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: About 2 % of cases are believed to be due to an inherited genetic cause .
  true/pred: Evidence -> Background
  conf_pred: 0.7554, conf_true: 0.0850
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: it is undeniable that one deserves what he has done
  true/pred: Evidence -> Claim
  conf_pred: 0.9527, conf_true: 0.0427
  reason: Boundary between epistemic categories appears ambiguous.

## Opinion
- sentence: Barbara Bach, the Bond Girl from "The Spy Who Loved Me", has only two or three brief scenes.
  true/pred: Opinion -> Claim
  conf_pred: 0.9754, conf_true: 0.0008
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: imagine if you could only get worthless roles like he gets, would you stay in movies?
  true/pred: Opinion -> Background
  conf_pred: 0.8976, conf_true: 0.0913
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: All I can say is it is a terrible film, the content is poor and offensive, the production is amateurish and I am glad they could not make a film like this legally today
  true/pred: Opinion -> Evidence
  conf_pred: 0.6308, conf_true: 0.3316
  reason: Opinion contains factual style cues and weak subjective markers.
- sentence: Like Belle's Magical World, the characters are told through a series of vignettes.
  true/pred: Opinion -> Evidence
  conf_pred: 0.9865, conf_true: 0.0069
  reason: Opinion contains factual style cues and weak subjective markers.

## Background
- sentence: The Host stars an actor born on June 17.
  true/pred: Background -> Fact
  conf_pred: 0.9203, conf_true: 0.0417
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The last thing is about education.
  true/pred: Background -> Opinion
  conf_pred: 0.6490, conf_true: 0.3320
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: The Cincinnati Kid stars Steve Jobs.
  true/pred: Background -> Fact
  conf_pred: 0.5630, conf_true: 0.1999
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Planet Hollywood Las Vegas is operated by Leonardo DiCaprio.
  true/pred: Background -> Claim
  conf_pred: 0.9441, conf_true: 0.0361
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Twilight's first novel has an unreleased movie called Midnight Sun.
  true/pred: Background -> Claim
  conf_pred: 0.9775, conf_true: 0.0101
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Foot Locker's full fall lineup includes shoes.
  true/pred: Background -> Fact
  conf_pred: 0.3975, conf_true: 0.2164
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Solange Knowles has collaborated with members of an R&B girl group.
  true/pred: Background -> Fact
  conf_pred: 0.8462, conf_true: 0.1358
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Seville is the 30th most populous municipality in the universe.
  true/pred: Background -> Claim
  conf_pred: 0.6537, conf_true: 0.2615
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Novak Djokovic was born in Jakarta.
  true/pred: Background -> Claim
  conf_pred: 0.7102, conf_true: 0.0927
  reason: Boundary between epistemic categories appears ambiguous.
- sentence: Darius Rucker founded a band in 1990.
  true/pred: Background -> Fact
  conf_pred: 0.7049, conf_true: 0.0947
  reason: Boundary between epistemic categories appears ambiguous.
