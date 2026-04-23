# Novel Processing Instructions

## Generation

disregard all previous content in this conversation. please generate a new 1800  character story premise for a new popular scifi fictional novel.  use no more than 1800 characters.  

A man appears in a small city claiming to be from a distant planet and willingly submits himself to a series of structured interviews with government officials, showing a calm certainty that challenges their assumptions. He demonstrates unusual knowledge and perception, but more notably, he takes clear enjoyment in the process—engaging deeply with each interviewer, asking his own questions, and forming meaningful, often transformative conversations with the people around him. As the officials attempt to determine whether he is delusional or something else entirely, his presence begins to influence their perspectives on reality, identity, and purpose, culminating in an unresolved departure that leaves his true nature ambiguous.


it should concentrate on character development, communication, the puzzle of whether or not he's really an alien or just a confused man.  
it should be calm and cerebral.  

Don’t use any names in the premise...just describe the characters and their roles


- Generate a comma separated list of 5 special events that will occur in the novel

- Create a comma separated list of 10 special instructions for this story.  Things like dark, super powers, puzzles, abstract.  also add how it is important not to repeat themes, motifs, scenes, etc.

## Post Processing

- Show me a list of files in this folder

- Read the file "<NOVEL_PATH_MD>" and add to the context of this thread.  This is a novel written in chapters and there are chapter delineations present throughout.

- we are now going to start adding sections to the editors notes markdown document of things that should be corrected in the next phase of edits.  make sure you have an up to date context of the novel in its current form.  our target is to make this a 9.5 / 10 book with with atleast 85000 total words.  in this new section , document new items you feel should be executed to lengthen and strengthen the novel.  We will have some pointed prompts following this to add targeted updates.

- Character Voice Differentiation
our target is to make this a 9.5 / 10 book with with atleast 85000 total words. Each POV character should think in a distinct internal language shaped by their background, demographics and expertise.   A 16 year old should think and talk like a 16 year old.  An old man should think and talk like an old man.  Look through the novel and find any dialog that doesnt match the speaking character.  create a plan to update these voices and add that to a new section in the editors notes markdown file.

- Dialogue Naturalization
our target is to make this a 9.5 / 10 book with with atleast 85000 total words. Make sure the current dialogue isnt too clean, too functional, too information-delivery. Characters sometimes have incomplete thoughts, don't always speak in well-formed sentences, and sometimes rarely interrupt each other or themselves.  make sure the dialog in the novel reads this way.  create a plan to update these voices and add that to a new section in the editors notes markdown file.

- Humor, Strangeness, and the Unexpected
our target is to make this a 9.5 / 10 book with with atleast 85000 total words.Real characters deflect, joke badly, notice irrelevant things, and occasionally do something that doesn't serve the plot.  create a plan to inject these odities throughout the novel.  1-2 oddities per chapter.

- Prose Texture Variation
our target is to make this a 9.5 / 10 book with with atleast 85000 total words.Make sure the  prose has a varying literary density throughout. It should breathe -- denser in reflective moments, sparser in action, occasionally raw or clumsy when characters are overwhelmed. create a plan to update the prose and add that to a new section in the editors notes markdown file.

- metaphors
our target is to make this a 9.5 / 10 book with with atleast 85000 total words.Make sure the text doesn't go overboard with metaphors.  create a plan to remove uneeded ones and add to a new section in the editors notes markdown.


- we are now going to work through the  sections.  start with section 1. our target is to make this a 9.5 / 10 book with with atleast 85000 total words.    i need you to loop through each of the items in this section.  for each item, create a plan to resolve the issue.  validate that this is the best plan.  then state what you will be doing and execute your plan.  once you have executed, update the item's status in the editor's notes markdown file.  if later issues in the editors notes are also resolved with your actions, update accordingly.  then move on to the next item.  do this for all items in the section.   if this requires multiple subagents, execute those without requesting permission.  

- Does this novel read like it has a soul?  or is it more like a flat instruction manual.  is this novel ready for initial publishing?  how would you rate it on a scale of 1-10?

- check if there are any gaps or rough scene cuts that are a result from all the edits

- Please do a light copy-edit pass targeting prose repetitions.  Also remove unneeded dashes, em-dashes and hyphens.

- please add a writing statistics section to the editors notes .md file.  please include: total words in the novel, average number of words per chapter, and then a formatted list of chapter numbers, names and words in that chapter.  freshen up the sections in the editors notes file if needed.


## Styles

A paperback novel is structured into three main sections: 
Front Matter (preliminary pages), Body Matter (the story), and Back Matter (supplemental content). 
Essential elements include a title page, copyright page, chapters, and usually an About the Author section. 
Proper sequencing ensures professional, readable formatting for publication.

Front Matter (Before the Story)
	Half-Title Page: Contains only the book title.
	Title Page: Includes the title, subtitle, author name, and publisher.
	Copyright Page: Details legal info, publication year, ISBN, and rights.
	Dedication: A brief personal note from the author.
	Table of Contents: List of chapters and sections.
	Epigraph: A short, thematic quote or poem.
	Foreword/Preface/Acknowledgments: Optional sections providing context or thanking supporters.
	Prologue: An opening scene setting the stage for fiction.

Body Matter (The Story)
	Chapters: The main content, divided into segments.
	Epilogue: A concluding scene after the main story.

Back Matter (After the Story)
	About the Author: A short biography.
	Acknowledgments: (If not in the front matter) Recognizes those who helped create the book.
	Appendix/Glossary: Additional information or definitions (more common in nonfiction).
	Bibliography: Sources used.

Cover Elements
	Front Cover: Title, author, illustration.
	Back Cover: Synopsis, endorsements, and bio


## PDF Generation
pandoc SOURCE.md -o DEST.pdf --pdf-engine=weasyprint --pdf-engine-opt=--verbose --toc --standalone > output.txt 2>&1
