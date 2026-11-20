# Scope: what the engine does not model

## Reach and frequency

The engine reads each channel's spend a period, and nothing about how many people the spend
reached or how often each saw it. A channel's response is a curve of its carried-over spend
(ADR 0064), so a plan moves money between channels and periods, never between reach and frequency
within a channel, and it cannot say what frequency a campaign should run at.

Of the tools mixdyn is compared with, Meridian models reach and frequency for the channels whose
data carry them. Until mixdyn does, a question about reach or frequency is Meridian's to answer,
and a channel bought by reach is read here by its spend alone.

It will be built if a user at launch asks for it and brings the reach and frequency data to test
it on; nothing is built for it before then.
