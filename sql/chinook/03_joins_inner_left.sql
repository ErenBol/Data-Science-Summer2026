-- 1. Using Track and Album, show each track with its album title.
SELECT al.Title as album_title, t.Name as track_name
FROM Album as al
INNER JOIN Track as t
ON al.AlbumId = t.AlbumId;

-- 2. Using Track, Album, and Artist, show each track with its album title and artist name.
SELECT t.Name as track_name, al.Title as album_title,  ar.Name as artist_name
FROM Track as t
INNER JOIN Album as al
ON al.AlbumId = t.AlbumId
JOIN Artist as ar
ON ar.ArtistId = al.ArtistId;


-- 3. Using Track and Genre, show each track with its genre name.
SELECT t.Name as track_name, g.Name as genre_name
FROM Track as t
INNER JOIN Genre as g
	ON g.GenreId = t.GenreId;


-- 4. Using Track, Album, Artist, and Genre, show each track with its album, artist, and genre.
SELECT t.Name as track_name, al.Title as album_title, ar.Name as artist_name, g.Name
FROM Artist as ar
INNER JOIN Album as al
	ON ar.ArtistId = al.ArtistId
INNER JOIN Track as t
	ON al.AlbumId = t.AlbumId
INNER JOIN Genre as g
	ON t.GenreId = g.GenreId;
	
-- 5. Using Track, Album, Artist, and Genre, find tracks whose genre is Rock.
SELECT t.Name as track_name, al.Title as album_title, ar.Name as artist_name, g.Name
FROM Artist as ar
INNER JOIN Album as al
	ON ar.ArtistId = al.ArtistId
INNER JOIN Track as t
	ON al.AlbumId = t.AlbumId
INNER JOIN Genre as g
	ON t.GenreId = g.GenreId
WHERE g.Name = 'Rock';

-- 6. Using Track, Album, and Artist, find tracks by AC/DC.
SELECT t.Name as track_name, al.Title as album_title, ar.Name as artist_name
FROM Artist ar
INNER JOIN Album al
	ON ar.ArtistId = al.ArtistId
INNER JOIN Track t
	ON al.AlbumId = t.AlbumId
WHERE ar.Name = 'AC/DC';

-- 7. Using Track, Album, Artist, and Genre, find tracks whose name contains Love.
SELECT t.Name as track_name, al.Title as album_title, ar.Name as artist_name, g.Name as genre_name
FROM Artist ar
INNER JOIN Album al
	ON ar.ArtistId = al.ArtistId
INNER JOIN Track t
	ON al.AlbumId = t.AlbumId
INNER JOIN Genre g
	ON t.GenreId = g.GenreId
WHERE t.Name LIKE '%Love%';

-- 8. Using Album and Track, show albums with their tracks if exists.
SELECT al.Title as album_title,  t.Name as track_name
FROM Album al
LEFT JOIN Track t
	ON al.AlbumId = t.AlbumId;
	
-- 9. Using Artist and Album, show artists with their albums.
SELECT ar.Name AS artist_name , al.Title as album_title
FROM Artist ar
LEFT JOIN Album al
	ON ar.ArtistId = al.ArtistId;


-- 10. Using Genre and Track, show genres with their tracks.    
SELECT g.Name AS genre_name, t.Name as track_name
FROM Genre g
LEFT JOIN Track t
	ON g.GenreId = t.GenreId;
