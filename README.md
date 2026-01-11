NBA Game Predictor

## How To Run
- Run model.py and click on the link produced in the terminal (locally-hosted webpage)
- The resulting webpage will give predictions for each matchup happening that night

## Disclaimers
- The trained models are from early December, so retrain the models with new data if wanting more accurate results.
- New data can be gathered using the Jupyter Notebooks under data_pipeline
- Models only factor in game statistics from 2022-23 season till the current season (early December)

## Data Source

NBA statistics data is retrieved using the [nba_api](https://github.com/swar/nba_api) 
Python package, which provides an interface to NBA.com's stats endpoints.

## Dependencies

- nba_api - NBA stats data retrieval
- pandas - Data manipulation

## Acknowledgments

- [nba_api](https://github.com/swar/nba_api) by Swar Patel for providing access to NBA statistics
- NBA.com for the underlying data