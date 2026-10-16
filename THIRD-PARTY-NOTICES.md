# Third-Party Notices

LMSupply is licensed under the MIT License (see `LICENSE`). It includes the third-party material listed below, each
under its own terms.

## CMU Pronouncing Dictionary

- Used by: `LMSupply.Synthesizer` (the built-in English phonemizer). The dictionary file is embedded in the assembly,
  unmodified and gzip-compressed (`src/LMSupply.Synthesizer/Data/cmudict/`).
- Source: <https://github.com/cmusphinx/cmudict>, `cmudict.dict` at commit `74790861f652b15e4ac49015a90074ad62a27690`.

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## NRL letter-to-sound rules

- Used by: `LMSupply.Synthesizer` (the built-in English phonemizer, for words the dictionary does not have). The rule
  table is in `src/LMSupply.Synthesizer/Core/English/LetterToSound.Rules.cs`.
- Source: H. S. Elovitz, R. W. Johnson, A. McHugh and J. E. Shore, «Automatic Translation of English Text to Phonetics
  by Means of Letter-to-Sound Rules», NRL Report 7948, Naval Research Laboratory, Washington, D.C., 21 January 1976
  (NTIS AD/A021 929) — a work of the United States Government — in the form published by John A. Wasser as «English to
  Phoneme Translation» (`english.c`, 15 April 1985), archived in the CMU Artificial Intelligence Repository
  (`areas/speech/systems/eng2phon/`).
- Terms: public domain. The published source states: «this is all public domain and I make no copyright claims on it.
  The part derived from the Naval Research Lab should be public anyway.»
