# Video File Naming Scheme Proposal

## Justification

Having a consistent naming schema aids in preserving what data was recorded
and allows for simple finding procedures as well as programmatic aids to allow
datasets to be loaded, and aggregrated without 
the need to specify their file system paths. Getting file system paths
right is a mild annoyance for programmers and a bit of a hindrance for less 
technical people, and given that a standard naming
system can alleviate many of these issues, let's attempt to adopt one.

## Aspects
 * Date
 * Chemicals Induced On The Inputs
 * Fly Descriptor (optional)
 * File Extension
### Date Formatting
ISO8601 Standard  
So:  
YYYYMMDD, or YYYY-MM-DD are both permissible. We need to preserve
the day on the file format, 
so let's keep that in there, even though it's not technically required to 
satisfy the standard, it makes a lot 
of sense to do for our purposes.
I would say that we should adopt the dashed version.

### Separator
According to: 
https://en.wikipedia.org/wiki/Filename#Comparison_of_filename_limitations, the 
fully portable file names, in general, allow for underscores in the filenames, as well 
as dashes. 
Let's adopt the protocol of using `-` for separating words and parts of dates, such as 
`wild-type`, and `_` for separating details of the file name such as 
the date. 

### Consideration of Input Gasses
If the input gasses are all let's say water vapor, then the file name
would look like  
`2022-06-06_water-vapor_wild-type.mp4`

If the left side only was water-vapor and the right side 
was let's say, air, I propose a standard
such that, the descriptor looks like the following:
`2022-06-06_L.water-vapor_R.air_wild-type.mp4`

For the corners, if need be specified, specify using
`UL`, `UR`, `LL`, `LR`, for upper left, upper right, 
lower left, lower right.

### Extension

Just use the most sensible extension, but ensure it's there.

### File Naming Constraints

To allow any programmatic parser to be deterministic and well-defined,
please refrain from using any non-alphanumeric character, in any place,
except the places outlined above.


