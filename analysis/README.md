# Linguistic Analysis

This directory contains code to calculate the lingusitic statistics presented in the paper and the supplementary materials

## Usage

Run python script from this directory:

python src/lingusitic_analysis data/model_output1 data/model_output2 data/model_output3


## Data

The expected format of the input data is the following:

I: [386603, 161846, 356290, 395097, 86285, 310714]
T: 86285
U: <sos> i have two orders of french fries and three hot dogs <eos>
P: <nohs>
A: <nohs>
R: ['i have two orders of french fries and three hot dogs', 'do you have the bowl of salad ?', 'i have the big bowl of salad', 'big bowl of salad']
H42: i have a white bowl with a wine gla * s
H1: do you have a picture of a brown dog on a table ?
H2: do you have the white bowl with the fork and the beer
H3: do you have a picture of a white bowl of a white bowl with the red and white bowl
H4: white bowl with the background

Where:
I: the ids of the images shown to the speaker
T: id of target image
U: utterance
P: any history if any of referring expressions to the same image
A: same as P, but without replacing rare vocabulary with <unk>
R: the reference chain
H*: the various seeds of model output: can be as few or many as desired.
