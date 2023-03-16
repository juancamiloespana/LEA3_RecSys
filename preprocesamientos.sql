
----procesamientos---

---crear tabla con usuarios con más de 50 libros leídos y menos de 1000

drop table if exists usuarios_sel;

create table usuarios_sel as 

select "User-Id" as user_id, count(*) as cnt_rat
from book_ratings
group by "User-Id"
having cnt_rat >50 and cnt_rat <= 1000
order by cnt_rat desc ;



---crear tabla con libros que han sido leídos por más de 50 usuarios
drop table if exists books_sel;



create table books_sel as select ISBN,
                         count(*) as cnt_rat
                         from book_ratings
                         group by ISBN
                         having cnt_rat >50
                         order by cnt_rat desc ;


-------crear tablas filtradas de libros, usuarios y calificaciones ----

drop table if exists ratings_final;

create table ratings_final as
select a."User-ID"as user_id,
a.ISBN as isbn,
a."Book-Rating" as book_rating
from book_ratings a 
inner join books_sel b
on a.ISBN =b.ISBN
inner join usuarios_sel c
on a."User-Id" =c.user_id;

drop table if exists users_final;

create table users_final as
select a."User-ID"as user_id,
a.Location as location,
a.Age as age
from users a
inner join usuarios_sel c
on a."User-Id" =c.user_id;

drop table if exists books_final;

create table books_final as
select a.ISBN as isbn,
a."Book-Title"  as book_title,
a."Book-Author" as book_author,
a."Year-Of-Publication" as year_pub,
a.Publisher as publisher,
a."Image-URL-S" as i_url
from books a
inner join books_sel c
on a.ISBN= c.ISBN;


---crear tabla completa ----

drop table if exists full_ratings ;

create table full_ratings as select 
a.*,
b.location,
b.age,
c.book_title,
c.book_author,
c.year_pub,
c.publisher,
c.i_url
 from ratings_final a inner join
 users_final b on a.user_id=b.user_id
 inner join books_final c on a.isbn=c.isbn;


