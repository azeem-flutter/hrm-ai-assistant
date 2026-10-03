# Sample test questions (Hinglish)

A graded list of example questions (easy to stress-test) useful for manually testing SQL-generation accuracy after any change to the prompt/RAG/model.

🟢 Level 1 — Easy
“Top 10 customers ko total sales ke basis par show karo.”

“Har category ki total sales aur total quantity show karo.”

“Har sales channel ke total orders aur total revenue show karo.”

“Mujhe 2025 mein place kiye gaye orders show karo.”

“Sabse expensive products ko unit price ke basis par descending order mein show karo.”

🟡 Level 2 — Basic Reasoning
“Har customer ke total orders aur total sales calculate karo aur highest sales wale customers ko pehle show karo.”

“Har category ka average unit price aur total quantity calculate karo.”

“Aise customers show karo jinhon ne 5 se zyada orders place kiye hain.”

“Har sales rep ki total sales calculate karo aur top 10 reps show karo.”

“Har province ki total sales calculate karo aur highest sales wale provinces ko pehle show karo.”

🟠 Level 3 — Moderate
“Top 100 customers by sales show karo aur unka total discount aur average discount percentage bhi show karo.”

“Aise products show karo jinki total quantity 100 se zyada hai aur total sales ko descending order mein arrange karo.”

“Har category mein total sales aur average discount percentage calculate karo aur categories ko sales ke basis par rank karo.”

“Aise customers find karo jinka total discount overall average discount se zyada hai.”

“Har city ke total sales aur average order value calculate karo aur highest revenue wali cities show karo.”

🔴 Level 4 — 3B ke liye challenging
“Aise customers find karo jinki total sales overall average customer sales se zyada hain aur average discount bhi overall average discount se zyada hai.”

“Aise products find karo jinki quantity apni category ki average quantity se zyada hai.”

“Har sales rep ka total revenue calculate karo aur sirf un reps ko show karo jinka revenue overall average sales rep revenue se zyada hai.”

“Aise customers identify karo jinhon ne kam az kam 3 orders kiye hain aur unki average order value overall average order value se zyada hai.”

“Aise orders find karo jinka total amount overall average order amount se zyada hai aur delivery 7 din se zyada leti hai.”

🔥 Level 5 — 3B Stress Test
Ye 3B ke liye difficult hain, lekin still realistic hain.

“Top customers identify karo jinki sales high hain aur average discount overall average discount se zyada hai. Saath mein total discount bhi show karo.”

“Aise customers find karo jinka total discount unki total sales ke 10% se zyada hai aur average discount percentage company average se higher hai.”

“Har category mein sabse zyada revenue generate karne wala product find karo.”

“Aise sales reps find karo jinka total revenue average sales rep revenue se zyada hai aur average discount company average se bhi zyada hai.”

“Aise customers identify karo jinhon ne multiple orders place kiye hain aur unka latest order amount unke apne average order amount se zyada hai.”