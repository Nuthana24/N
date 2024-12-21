from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configure the SQLAlchemy part of the app instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///items.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Secret key for session management
app.secret_key = os.urandom(24)

# Create the SQLAlchemy database instance
db = SQLAlchemy(app)

# Ensure the `static/images` directory exists
image_dir = os.path.join(os.getcwd(), "static", "images")
os.makedirs(image_dir, exist_ok=True)

# Define the User model for sign-up and login functionality
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

# Define the Item model for storing found/lost items
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    contact_info = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(10), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "location": self.location,
            "contact_info": self.contact_info,
            "status": self.status,
            "image_path": self.image_path,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize the database
with app.app_context():
    db.create_all()

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to extract features from an image
def extract_features(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_img(image_path, target_size=(224, 224))  # Resize image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features

# Function to find similar items
def find_similar_items(new_image_path, existing_items):
    full_new_image_path = os.path.join(image_dir, os.path.basename(new_image_path))
    if not os.path.exists(full_new_image_path):
        raise FileNotFoundError(f"New image not found: {full_new_image_path}")

    new_image_features = extract_features(full_new_image_path)
    similarities = []

    for item in existing_items:
        full_existing_path = os.path.join(image_dir, os.path.basename(item.image_path))
        if os.path.exists(full_existing_path):
            existing_features = extract_features(full_existing_path)
            similarity = cosine_similarity(new_image_features, existing_features)
            similarities.append((item, similarity[0][0]))

    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Route: Home page (display items)
@app.route("/")
def index():
    items = Item.query.all()
    for item in items:
        print(item.image_path)  # Check the paths being retrieved
    status_filter = request.args.get("status", "lost")  # Default to 'lost' if no status is provided
    items = Item.query.filter_by(status=status_filter).all()  # Filter items by status
    return render_template("index.html", items=items, status=status_filter)


# Route: Report lost or found item
@app.route("/report", methods=["GET", "POST"])
def report():
    if request.method == "POST":
        title = request.form["title"]
        category = request.form["category"]
        description = request.form["description"]
        location = request.form["location"]
        contact_info = request.form["contact_info"]
        status = request.form["status"]

        # Handle image upload
        image = request.files["image"]
        if image:
            filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{image.filename}"
            image_path = os.path.join("images", filename)  # Save relative to 'static/'
            image.save(os.path.join(image_dir, filename))
        else:
            return "Image is required", 400

        # Save to database
        new_item = Item(
            title=title, category=category, description=description,
            location=location, contact_info=contact_info, status=status,
            image_path=image_path
        )
        db.session.add(new_item)
        db.session.commit()

        # Find similar items
        items = Item.query.all()
        similar_items = find_similar_items(image_path, items)

        filtered_similar_items = [(item, score) for item, score in similar_items if item.id != new_item.id and score > 0.7]
        
        if filtered_similar_items:
            # Redirect to index page and show similar items
            return render_template("index.html", items=items, similar_items=filtered_similar_items)

        return redirect(url_for("index"))
    return render_template("report.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

# Route: Admin panel (manage items)
@app.route("/admin")
def admin():
    # Check if admin is logged in
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    items = Item.query.all()
    similar_items = {}

    # Find similar items for each product
    for item in items:
        similar_results = find_similar_items(item.image_path, items)
        filtered_results = [(similar_item, score) for similar_item, score in similar_results 
                            if similar_item.id != item.id and score > 0.7]
        if filtered_results:
            similar_items[item.id] = filtered_results

    return render_template("admin.html", items=items, similar_items=similar_items)

# Route: Admin login
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check admin credentials (you can set admin credentials here)
        if username == "admin" and password == "password":  # Use real credential management in production
            session['admin_logged_in'] = True
            return redirect(url_for("admin"))
        else:
            return "Invalid credentials. Please try again."

    return render_template("admin_login.html")

# Route: Admin logout
@app.route("/admin_logout")
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for("index"))

# Route: Edit an item
@app.route("/edit/<int:item_id>", methods=["POST"])
def edit_item(item_id):
    item = Item.query.get_or_404(item_id)

    # Update item details
    item.title = request.form["title"]
    item.category = request.form["category"]
    item.description = request.form["description"]
    item.location = request.form["location"]
    item.contact_info = request.form["contact_info"]
    item.status = request.form["status"]

    # Handle image upload (optional)
    if "image" in request.files:
        image = request.files["image"]
        if image.filename:
            filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{image.filename}"
            image_path = os.path.join(image_dir, filename)
            image.save(image_path)
            item.image_path = os.path.join("images", filename)

    db.session.commit()
    return redirect(url_for("admin"))

# Route: Delete an item
@app.route("/delete/<int:item_id>", methods=["POST"])
def delete_item(item_id):
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for("admin"))

# API: Get all items as JSON
@app.route("/api/items")
def get_items():
    items = Item.query.all()
    return jsonify([item.to_dict() for item in items])

# API: Get a single item as JSON
@app.route("/api/item/<int:item_id>")
def get_item(item_id):
    item = Item.query.get_or_404(item_id)
    return jsonify(item.to_dict())

# Sign-up route
@app.route("/sign_up", methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Hash the password before saving
        hashed_password = generate_password_hash(password)

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists. Please choose a different username."

        # Create a new user
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("sign_up.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    next_page = request.args.get('next')  # Get the 'next' parameter

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check user credentials
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['logged_in'] = True

            # Redirect to the 'next' page if specified, else to index
            return redirect(url_for(next_page)) if next_page else redirect(url_for("index"))
        else:
            return "Invalid credentials. Please try again."

    return render_template("login.html", next=next_page)

if __name__ == "__main__":
    app.run(debug=True)
