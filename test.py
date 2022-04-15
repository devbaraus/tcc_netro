# %%
import eyed3

# %%
audiofile = eyed3.load(
    "/src/spotify/mp3/alt-J/An Awesome Wave/Breezeblocks - alt-J.mp3")

# %%
audiofile.tag.artist
audiofile.tag.album
audiofile.tag.genre.name
audiofile.tag.title
# %%
