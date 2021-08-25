# 3dreconstruction
Hej!

Texten är osammanhängande, dels som helhet men också i kapitel och individuella meningar. Det är tydligt att du inte nyttjat top-down och det går sällan att bygga en bra rapport bottom-up. Kvaliteten i arbetet utöver dokumentationen kommer inte fram då den helt skyms av en undermålig presentation. Nu kanske din idé är att jag ska gå in och kolla på de individuella delarna/detaljerna och sen gör du en sammansättning av dessa. Det är svårt att göra och brukar heller inte fungera. Bättre att du gör den helt ”klar” först.

Generellt:
Din struktur och presentation saknar flöde. Allt ska sitta ihop men grundläggande skrivregler gäller.
Det är författarna som gjort arbetet, referera till dem och använd inte referensen som ett objekt.
Varför kommer kapitel 2 före 3 och 4?
Kap – Kommentar
1 – Introduktionen verkar gå in på metoden, är det tanken, och i så fall varför?
1 – Vad är ArUco? Du måste presentera begrepp och förkortningar innan du använder dem. Om det inte passar att göra i introduktionen passar inte heller att omnämna det i introduktionen.
1 – Jag tycker att din introduktion är ofokuserad. Tips (som jag har sagt tidigare): sammanfatta varje sektion med en mening. Bygg sedan ut detta och studera flödet. Kom ihåg att det består av trattar! Du pratar om features och annotation innan du etablerat området; inte bra flöde.
1 – Väldigt konstiga meningar: ” The methods used in this paper attempt to compare the output data from how with a human in
the image domain of the dataset and 3D reconstruction for how would react to a human laying
on the floor in the global domain”?!
2 – Ser inget om koncept, endast antaganden på icke introducerade koncept. Först motivering, sen specificering. Du måste bygga en röd tråd (en historia)!
2.4 – ”previously indexed” är det något som utförs senare eller är det bara ett antagande och en förutsättning?
2.4 – ArUco Markers har väl inte egenskaper unika mot andra konstgjorda markörer eller absoluta deskriptorer? Att jämföra med stereomatchning av okända pixlar känns märkligt, som selfie (äpplen) och satellitbild (päron) för kartläggning av din tomt. Vidare har du inte resonerat över lämpligheten med strategiskt utplacerade konstgjorda markörer.
2.5 – Otydligt om du menar en kamera som flyttas runt eller flera kameror. Om det senare bör det tydliggöras om de är fasta eller mobila.
2.8 – Är ”pose quiver” ett eget begrepp?
2.9 – Hur definierar du Origin och hur kom roboten/kameran dit?
2.10 – ” If that is done for a feature in at least two cameras, the 3D position can be derived”: Position för vad? Kameran/markören/personen?
2.11 – Är markörerna kopplade till mänskliga attribut?

Längre har jag inte hunnit och dessa kommentarer är ingalunda uttömmande. Jag har försökt kommentera på en konceptuell nivå för att inte detaljpåverka, det finns mycket mer att kommentera på detaljerna.


Är alla bilder egengjorda eller open source?
Med vänlig hälsning,
Fredrik

# Report status | project:xjob.report
* [ ] 1 intro aruco  #3c5f422d
* [ ] 1 fix introduction  #f61d11be
* [ ] 1 Konsept is not inroduced  #b82b0ea3
* [ ] Previosly indexed, later or an assumption  #9f964b1d
* [ ] Initial thoughts, floating camera or fixt camera?  #4ff2c80e
* [ ] inital thoughts pose quiver ref missing  #937ec105
* [ ] inital thoughts How is origin defined  #824c97de
* [ ] inital thoughts 'If that is done for a feature in atleast two cameas' is inplistit  #47cdf95c
* [ ] inital thoughts is marker connected to human attributs?  #6db134b1
