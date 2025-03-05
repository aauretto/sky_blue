# Planning

## currently Implemented
  - minibatch of files within one band based on time stamp
## Need to implement
  - iterate over all batches of files within the same band
  - iterate over all bands
    - Bands we think will be most helpful for turbulence detection
      - 8 (wavelength 6.2): upper level water vapor
      - 9 (wavelength 6.9): mid level water vapor
      - 10 (wavelength 7.3): low level water vapor
      - 13 (wavelength 10.3): clean long wave infared window (storm detection, could cover features others don't include)
      - 14 (wavelength 11.2) infrared long wave
      - 15 (wavelength 12.3) dirty window
    - Bands used for identifying clear skies (preceded by wavelength)
      - .64 (band 2), 1.38 (band 4), 1.61 (band 5), 7.3 (band 10), 8.4 (band 11), 11.2 (band 14), 12.2 (band 15)
      - Overlap:
        - Band 10
        - Band 14
        - Band 15
      - Not overlap:
        - Band 2
        - Band 4
        - Band 5
        - Band 11

## Resources

[Band Source](https://rammb2.cira.colostate.edu/training/visit/quick_reference/#tab17)