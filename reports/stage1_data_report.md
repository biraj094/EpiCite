# Stage 1 Data Report (Repaired v2)

## Source Notes
- IBM primary candidate (Claim+Evidence+Context) was not publicly reachable in this environment.
- IAM is removed by design: stance +/-1 is topic stance, not Claim/Evidence semantic type.
- UKP/AAEC: ArgumentAnnotatedEssays-2.0 (MajorClaim/Claim/Premise + Non-Argument sentence extraction).
- FEVER source used: copenlu/fever_gold_evidence (REFUTES claims + SUPPORTS claims + SUPPORTS evidence + NEI claims).
- WikiSQE is removed from Stage 1 training data to avoid leakage into Stage 2.
- Opinion source used: stanfordnlp/imdb with TextBlob subjectivity > 0.7.
- Opinion dual filter: TextBlob > 0.7 AND (first-person OR opinion lexicon).

## Source Original Label Distribution

### aaec_ukp
- Premise: 3832
- Claim: 1506
- Non-Argument: 1326
- MajorClaim: 751

### fever
- SUPPORTS: 124133
- NOT ENOUGH INFO: 79246
- REFUTES: 56872

### imdb
- 0: 8994

## Mapping Rules
- AAEC MajorClaim/Claim -> Claim
- FEVER REFUTES claim -> Claim
- AAEC Premise -> Evidence
- AAEC Non-Argument -> Background (or Opinion if first-person) 
- FEVER SUPPORTS claim -> Fact
- FEVER SUPPORTS evidence sentence -> Evidence
- FEVER NOT ENOUGH INFO claim -> Background
- IMDB subjective sentence (TextBlob > 0.7) -> Opinion

## Cleaning Stats
- input_rows: 35200
- dropped_duplicates: 15
- dropped_length: 0
- dropped_non_english: 315

## Final 5-Class Distribution
- Claim: 5997 (0.3409)
- Fact: 3397 (0.1931)
- Evidence: 3400 (0.1933)
- Opinion: 2400 (0.1364)
- Background: 2399 (0.1364)

## Split Sizes
- train: 14074
- val: 1759
- test: 1760

## Length Statistics
- FEVER REFUTES sampled claim length mean: 8.69
- FEVER REFUTES sampled claim length median: 8.00
- IMDB candidates after dual filter: 9000
- IMDB baseline from previous run: 4190

## Random Samples (15 per class)

### Claim
- [aaec_ukp/Claim] Even if you are not working in a group, you still need to take others' criticism
- [fever/REFUTES_claim] John II of Portugal was only ever the king of France.
- [aaec_ukp/Claim] If these companies were obliged to pay to clean up the air pollution, they would at least make an effort to reduce the amount of air pollution they cause
- [fever/REFUTES_claim] Tom Hardy refused to appear in Inception (2010).
- [aaec_ukp/Claim] the measure cannot be adopted, otherwise those companies will go into liquidation
- [fever/REFUTES_claim] Afghanistan is the source of only the Hotak dynasty.
- [aaec_ukp/Claim] a single international language serves an economic purpose
- [aaec_ukp/MajorClaim] government should give priorities to invest more money on the basic social welfares such as education and housing instead of subsidizing arts relative programs
- [fever/REFUTES_claim] Al Capone was not prosecuted for a federal crime.
- [fever/REFUTES_claim] Season 2 of Fargo is a sequel to the events in the first season.
- [fever/REFUTES_claim] Aaliyah was raised outside Michigan.
- [fever/REFUTES_claim] Cleopatra's legacy has ceased.
- [aaec_ukp/MajorClaim] there will be a decrease in cars use in the future
- [aaec_ukp/Claim] growing of cheap flights really have negative impact on environment
- [aaec_ukp/Claim] Universities should accept equal numbers of students of both genders
- [fever/REFUTES_claim] Ancient Algeria has known only a couple empires and dynasties.
- [fever/REFUTES_claim] Lana Del Rey released Summertime Sadness in 2013.
- [fever/REFUTES_claim] Alan White has no experience as an Ambassador.
- [fever/REFUTES_claim] Bohemian Rhapsody was a limited success.
- [fever/REFUTES_claim] The Right Stuff was unable to cast Sam Shepard.

### Fact
- [fever/SUPPORTS_claim] Jacqueline Kennedy Onassis was admired.
- [fever/SUPPORTS_claim] Steam is the gaseous state of water.
- [fever/SUPPORTS_claim] Stars Are Blind was written by Ralph McCarthy and others.
- [fever/SUPPORTS_claim] The New Adventures of Old Christine is a sitcom made in America.
- [fever/SUPPORTS_claim] Lee Min-ho first acquired widespread fame in Korea and parts of Asia with his role as Gu Jun-pyo in Boys Over Flowers.
- [fever/SUPPORTS_claim] 25 was released in 2015.
- [fever/SUPPORTS_claim] Marie Curie was a chemist.
- [fever/SUPPORTS_claim] Wyatt Earp worked to extract minerals.
- [fever/SUPPORTS_claim] Ivan Lendl has won two WCT Finals title.
- [fever/SUPPORTS_claim] In 2011 Lighting Point was filmed in the Gold Coast.
- [fever/SUPPORTS_claim] Bob Riley has a full name.
- [fever/SUPPORTS_claim] Brad Pitt portrayed a cowboy hitchhiker.
- [fever/SUPPORTS_claim] Alexander Lukashenko has been in office since 20 July.
- [fever/SUPPORTS_claim] Furia is adapted from a short story by an essayist.
- [fever/SUPPORTS_claim] Linkin Park is a group.

### Evidence
- [fever/SUPPORTS_evidence] `` Never Wanted Nothing More '' is a song written by Ronnie Bowman and Chris Stapleton , and recorded by American country music artist Kenny Chesney .
- [fever/SUPPORTS_evidence] Before the album , Jay-Z had released collaborations with The Roots and R. Kelly , and Linkin Park had collaborated with various artists on their remix album Reanimation .
- [fever/SUPPORTS_evidence] Wilde 's parents were successful Anglo-Irish , Dublin intellectuals .
- [fever/SUPPORTS_evidence] Jack Lowden -LRB- born 2 June 1990 -RRB- is a Scottish stage , television , and film actor .
- [fever/SUPPORTS_evidence] The film stars Al Pacino , Diane Keaton , Talia Shire , and Andy García , and features Eli Wallach , Joe Mantegna , George Hamilton , Bridget Fonda , and Sofia Coppola .
- [fever/SUPPORTS_evidence] In `` Post Mortem , '' he left the Diagnostic Team after realizing he was in the same position as he was 10 years earlier , unlike all of the other former members of the team .
- [fever/SUPPORTS_evidence] Following his film debut in the drama Taps -LRB- 1981 -RRB- and a diverse range of film roles in the 1980s , including Fast Times at Ridgemont High -LRB- 1982 -RRB- , Penn garnered critical attention for his roles in the crime dramas At Close Range -LRB- 1986 -RRB- , State of Grace -LRB- 1990 -RRB- , and Carlito 's Way -LRB- 1993 -RRB- .
- [fever/SUPPORTS_evidence] He began his career as a junior actor and went on to establish himself as one of the biggest stars in Bollywood .
- [aaec_ukp/Premise] Individuals purchase their clothes according to the latest fashion trends, which may result in the phenomenon that people would wear the same style of clothes
- [aaec_ukp/Premise] Many hardworking people have diseases, like the burn-out syndrome
- [fever/SUPPORTS_evidence] In 1922 , she made the choice to resign her position at the library to spare more time for teaching .
- [fever/SUPPORTS_evidence] Stolen Babies are an American experimental rock band consisting of vocalist/accordionist Dominique Lenore Persi , bassist/guitarist Rani Sharone , and drummer Gil Sharone .
- [fever/SUPPORTS_evidence] Batgirl is the name of several fictional superheroes appearing in American comic books published by DC Comics , depicted as female counterparts to the superhero Batman .
- [fever/SUPPORTS_evidence] Villa Park also hosted the 2012 FA Community Shield , as Wembley Stadium was in use for the final of the Olympic football tournament .
- [fever/SUPPORTS_evidence] He played Pavel Chekov in three Star Trek films , including the 2009 reboot film of the same name , along with the sequels , Star Trek Into Darkness and the posthumously released Star Trek Beyond -LRB- 2016 -RRB- .

### Opinion
- [imdb/review_label_0] He gurns and gesticulates excessively and looks rather ridiculous as a result.
- [imdb/review_label_0] My copy may have been affected but I was disappointed with the lack of menu screen for the DVD.
- [imdb/review_label_0] This movie is truly boring.
- [imdb/review_label_0] Like the screenplay, the look of the film is joyless and at times aesthetically barren and surreal.
- [imdb/review_label_0] Cécile's child is bisexual and is bitten by dogs (loyalty) once he meets his boyfriend, whereas the girl he lives with seems to be sick (of that?).
- [imdb/review_label_0] Glenn Ford, as the expert called upon to defuse the bomb, is given awful writing material to work with.
- [imdb/review_label_0] I read about this movie and it sounded so awful that I had to see it, and my gosh, I can smell it in St Louis.
- [imdb/review_label_0] Julie loves him nonetheless and continually makes excuses for him, which only seems to make him more abusive.
- [imdb/review_label_0] The actor who played 'Scarecrow' was amazing, I will say that.
- [imdb/review_label_0] If i would list all the flaws in the movie , this review would take me a lot of sentences.( very funny flaws, because of being that bad though) You got to be Amazed with the skill of the commandos assigned to rescue the plane.
- [imdb/review_label_0] Now for Rick dean lol, in Carnosaur 2 I thought he fit the role pretty well and wasn't really annoyed by him, now in Carnosaur 3 wow they placed him as an elite soldier.
- [imdb/review_label_0] It's inconceivable how someone could possibly come up with something so stupid and think it was entertaining.
- [imdb/review_label_0] In scenes that required immense tension and buildup, it felt like necessary frames were cut.
- [imdb/review_label_0] Pretty strange and pretty awful.
- [imdb/review_label_0] From that point, we only wait until she starts having crises.
- [imdb/review_label_0] So the mother's brilliant idea is to call the jerry springer show as well as getting it on with her daughter's boyfriend.
- [imdb/review_label_0] While I fought boredom here watching jokes that were silly and stupid, brutally dated and brutally bad, the more recent parody had me laughing out loud.
- [imdb/review_label_0] I only watched this because it was directed by Lucio Fulci and featured Claudio Cassinelli, an actor I like.
- [imdb/review_label_0] Something I'm noticing as well is that the pacing isn't really suspenseful in a typical way.
- [imdb/review_label_0] We're supposed to find this kids likable and nice.

### Background
- [fever/NEI_claim] Lily James killed Countess Natasha Rostova
- [fever/NEI_claim] John Mayer voted on a Grammy Award.
- [aaec_ukp/Non-Argument] Thus, both government and individual should take steps to diminish the consequences involved.
- [fever/NEI_claim] Walt Disney founded a company, Disney Movie Company, headquartered in Burbank.
- [fever/NEI_claim] Yves Saint Laurent was celebrated in 1961.
- [aaec_ukp/Non-Argument] Capital punishment; 51% countries have polished death penalty "Capital punishment or the death penalty is a legal process whereby a person is put to death by the state as a punishment for a crime.
- [fever/NEI_claim] Lightning Point was filmed in a live studio in the Gold Coast in 2011.
- [fever/NEI_claim] Blue Jasmine is about a rich Manhattan socialite who became a Christian.
- [fever/NEI_claim] Anne Hathaway had a role in a graveyard.
- [fever/NEI_claim] The Offspring's third jazz studio album is Smash.
- [fever/NEI_claim] Dhoom 3 was spearheaded by Vijay Krishna Acharya.
- [aaec_ukp/Non-Argument] Computer has negative effects to children Nowadays, thanks to the development of technology, computer is now indispensable to life.
- [fever/NEI_claim] Sally Field was in Wonder Woman.
- [fever/NEI_claim] Pierce Brosnan was in Dante's Peak and gained criticism.
- [fever/NEI_claim] Caryn Mandabach acted in A Different World.

## Potentially Weak Mapping Rules
- fever_refutes_claim_to_claim: REFUTES claims can include noisy or ambiguous claims.
- imdb_subjective_sentence_to_opinion: TextBlob subjectivity can mis-score rhetorical factual sentences.
- aaec_non_argument_first_person_to_opinion: first-person heuristic may over-generate Opinion.
- Suspicious sample count (rules above): 6189
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Second Punic War ended in 222 BC.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: L.A. Law is an American television commercial.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Taylor Sheridan has only ever portrayed Tony Hale.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Charles, Prince of Wales was the second grandchild of King George VI.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Jing Tian is completely Mexican.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Kitti's hog-nosed bat's middle name is Craseonycteris thonglongyai.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Bangladesh is not in the Bengal region.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Samuel L. Jackson has not appeared in the Marvel Cinematic Universe.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Antarctica contains the North Pole.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Haiti has not had a revolution
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Paul Pogba ended his career in 2011.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: John Stewart was the first African American superhero to be featured in a film.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: KJ Apab has avoided acting entirely.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Gotham City Sirens contained no art.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Slovenia has a sparse river network.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Antigua and Barbuda was the birthplace of Christopher Columbus.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Henry III assumed the throne when he was 2 years old.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: School 2013 has no role for Jang Nara.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Seattle is the largest inland city in the Pacific Northwest region of North America.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Claire Danes did not receive an Emmy Award.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Beatles was a pest infestation in the 1970s.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Mike Tyson's only match against Evander Holyfield was in 1990.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Wolfgang Amadeus Mozart was a wrestler.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Closer's final season was its ninth season.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Election (1999 film) was only written by Jim Taylor.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Beijing is rarely the nation's educational center.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Major League Soccer playoffs leads to the NBA finals.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Cars 3 isn't the third Cars movie Brian Fee has worked on.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Elementary passed up Jon Michael Hill for a role in its cast.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Kurt Sutter is the star of Sons of Anarchy.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Ramadan is the ninth month of the Jewish calendar.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Leonardo Bonucci doesn't play Italian football.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Red Hot Chili Peppers' original line-up had yet to acquire guitarist Hillel Slovak.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Connie Britton was snubbed in all Golden Globe Award categories pertaining to her performance on Nashville.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Foo Fighters formed in space.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Adrien Broner was ranked third best boxer in the world.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Platinum Dunes had nothing to do with A Nightmare on Elm Street.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Legend of Tarzan (film) was released in 2013.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: None of Lake Powell is in Utah.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Manchester City F.C. is a English football stadium.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Minnesota Vikings are based in Athens, Georgia.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Little Big Shots is an American side dish.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Wolfman (2010 film) was directed by Steven Spielberg.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Azerbaijan's official nickname is Azerbaijan People's Republic.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The principal photography of Vantage Point (film) began on the day of 19th.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Diane Ruggiero writes for Brooklyn Nine Nine.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Outlander (TV series) failed to be renewed for a fourth season.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: The Democratic Republic of the Congo is a city-state.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Mother Teresa was canonised on June 4 2016.
- [fever/REFUTES_claim -> Claim] reason=fever_refutes_claim_to_claim: Donald Trump's only middle name is Hilary.

## Data Leakage Check
- Stage 1 train sentences were matched by exact normalized string against WikiSQE_experiment/all (train+val+test).
- WikiSQE rows checked: 2487996
- Stage 1 train unique sentences: 14074
- Exact overlap count: 0
- Result: No overlap detected.