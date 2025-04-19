from supabase import create_client, Client

# Supabase credentials
url = "https://byunlkvjaiskurdmwese.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ5dW5sa3ZqYWlza3VyZG13ZXNlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MTM4NDg3MCwiZXhwIjoyMDU2OTYwODcwfQ.2OeQdxSb3VLyEfZVY5SO45nz2WM7YzlS3lgTNhLtnbw"  # Use your actual service role key (not anon key)

supabase: Client = create_client(url, key)

# Your new image URL
new_image_url = "https://byunlkvjaiskurdmwese.supabase.co/storage/v1/object/sign/images/ChatGPT%20Image%20Apr%2018,%202025,%2010_59_37%20PM.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJpbWFnZXMvQ2hhdEdQVCBJbWFnZSBBcHIgMTgsIDIwMjUsIDEwXzU5XzM3IFBNLnBuZyIsImlhdCI6MTc0NTA3Nzg2NywiZXhwIjoxNzc2NjEzODY3fQ.rUSlBPCI0YWLgq1JAfIrR2nLbAOPa-EF30cFyveDkZ4"

# Update all rows (filtering by condition that always applies)
response = supabase.table("destinations").update({
    "images": new_image_url
}).neq("id", -1).execute()  # This condition always returns all rows

# Print result
print("✅ Updated image URL for all rows.")
