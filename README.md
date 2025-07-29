> ### Please consider giving back to the community if you have benefited from this work.
>
> If you've **benefited commercially from this work**, which we've poured significant effort into and released under permissive licenses, we hope you've found it valuable! While these licenses give you lots of freedom, we believe in nurturing a vibrant ecosystem where innovation can continue to flourish.
>
> So, as a gesture of appreciation and responsibility, we strongly urge commercial entities that have gained from this software to consider making voluntary contributions to music-related non-profit organizations of your choice. Your contribution directly helps support the foundational work that empowers your commercial success and ensures open-source innovation keeps moving forward.
>
> Some suggestions for the beneficiaries are provided [here](https://github.com/the-secret-source/nonprofits). Please do not hesitate to contribute to the list by opening pull requests there.

---



# Divide and Remaster v3

Divide and Remaster v3 is a multilingual rework of the Divide and Remaster v2 dataset by Pétermann et al. 

The major changes from DnR v2 are as follows:
- the dialogue stem now contains content from more than 30 languages across various language families;
- speech, vocals, and/or vocalizations have been removed from the music and effects stems;
- loudness and timing parametrization have been adjusted to approximate the distributions of real cinematic content;
- the mastering process now preserves relative loudness between stems and approximates standard industry practices.

**See [wiki](https://github.com/kwatcharasupat/divide-and-remaster-v3/wiki) for instructions on using this dataset**.

## Model

For the model trained on DnR v3, go [here](https://github.com/kwatcharasupat/bandit-v2)

## Recreating the dataset

The source code for recreating the dataset in in `dnr-v3`.

## License

Divide and Remaster v3 is released under the CC BY-SA 4.0 license. See wiki for full license information of each source dataset.
