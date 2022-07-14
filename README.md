# This is my second project. I made it myself.

## Cost prediction of cryptocurrencies. The data is taken from the competition [kaggle](https://www.kaggle.com/c/g-research-crypto-forecasting/overview)

#### Over $40 billion worth of cryptocurrencies are traded every day. They are among the most popular assets for speculation and investment, yet have proven wildly volatile. The dataframe consists information about past transactions. For example: Bitcoin, Ethereium, etc.
#### Task - predict the future profits of cryptocurrencies using machine learning.

* ### Dataframes:

1. train.csv - train dataframe

    * timestamp - A timestamp for the minute covered by the row.
    * Asset_ID - An ID code for the cryptoasset.
    * Count - The number of trades that took place this minute.
    * Open - The USD price at the beginning of the minute.
    * High - The highest USD price during the minute.
    * Low - The lowest USD price during the minute.
    * Volume - The number of cryptoasset units traded during the minute.
    * VWAP - The volume weighted average price for the minute.
    * Target - Residual log-returns for the asset over a 15 minute horizon.


2. asset_details.csv - dataframe with additional information

    * Asset_ID - An ID code for the cryptoasset.
    * Weight - Weight of each assets used to weight its relative importance.
    * Asset_Name - The name of cryptocurrency.