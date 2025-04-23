import os
import numpy as np
import matplotlib.pyplot as plt
import json
import io
import base64

def plot_sentiment_trajectory(profile, output_dir=None, return_base64=False):
    """
    Plot the sentiment trajectory of the book.
    
    Args:
        profile (dict): Book profile data
        output_dir (str): Directory to save plots to
        return_base64 (bool): Whether to return base64 encoded images instead of saving
        
    Returns:
        dict: Base64 encoded images if return_base64 is True
    """
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    trajectory = profile["sentiment_trajectory"]["smoothed"]
    polarity = trajectory["polarity"]
    emotions = trajectory["emotions"]
    
    base64_images = {}
    
    # Plot polarity
    plt.figure(figsize=(12, 6))
    plt.plot(polarity)
    plt.title("Sentiment Polarity Throughout the Book")
    plt.xlabel("Chapter")
    plt.ylabel("Polarity (-1 to 1)")
    plt.grid(True, alpha=0.3)
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        base64_images['polarity'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    elif output_dir:
        plt.savefig(os.path.join(output_dir, "polarity_trajectory.png"))
        plt.close()
    
    # Plot emotions
    N = 10
    avg_intensity = {e: np.mean(v) for e, v in emotions.items()}
    top_emotions = sorted(avg_intensity.items(), key=lambda x: x[1], reverse=True)[:N]
    top_emotions = {e: emotions[e] for e, _ in top_emotions}
    
    plt.figure(figsize=(12, 8))
    for emotion, values in top_emotions.items():
        plt.plot(values, label=emotion.capitalize())
    
    plt.title("Emotional Trajectory Throughout the Book")
    plt.xlabel("Chapter")
    plt.ylabel("Emotion Intensity")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        base64_images['emotions'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    elif output_dir:
        plt.savefig(os.path.join(output_dir, "emotion_trajectory.png"))
        plt.close()
    
    # Create stacked area chart
    plt.figure(figsize=(14, 8))
    used_emotions = {e: vals for e, vals in emotions.items() if np.any(np.array(vals) > 0)}
    
    if used_emotions:
        avg_intensities = {e: np.mean(vals) for e, vals in used_emotions.items()}
        top_n = 10
        top_emotions = sorted(avg_intensities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        selected_emotions = [e for e, _ in top_emotions]
        emotions_array = np.array([used_emotions[e] for e in selected_emotions])
        labels = [e.capitalize() for e in selected_emotions]
        
        # Plot
        plt.stackplot(range(len(emotions_array[0])), emotions_array, labels=labels, alpha=0.8)
        plt.title("Emotional Composition Throughout the Book")
        plt.xlabel("Chapter")
        plt.ylabel("Proportion")
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if return_base64:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            base64_images['emotion_composition'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        elif output_dir:
            plt.savefig(os.path.join(output_dir, "emotion_composition.png"))
            plt.close()
    
    if return_base64:
        return base64_images
    elif output_dir:
        print(f"Plots saved to {output_dir}")
        return None

def plot_character_emotions(profile, output_dir=None, return_base64=False):
    """
    Plot the emotions associated with main characters.
    
    Args:
        profile (dict): Book profile data
        output_dir (str): Directory to save plots to
        return_base64 (bool): Whether to return base64 encoded images instead of saving
        
    Returns:
        str: Base64 encoded image if return_base64 is True
    """
    character_emotions = profile["character_emotions"]
    if not character_emotions:
        print("No character emotions to plot")
        return None
    
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 characters by total emotion intensity
    top_chars = sorted(
        character_emotions.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:5]
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.15
    
    # Set positions on X axis
    emotions = list(next(iter(character_emotions.values())).keys())
    r = np.arange(len(emotions))
    
    # Create bars
    for i, (char, char_emotions) in enumerate(top_chars):
        emotion_values = [char_emotions[e] for e in emotions]
        ax.bar(r + i * barWidth, emotion_values, width=barWidth, label=char)
    
    # Add labels
    plt.xlabel('Emotions', fontweight='bold')
    plt.ylabel('Intensity', fontweight='bold')
    plt.title('Emotions Associated with Main Characters')
    plt.xticks([r + barWidth * 2 for r in range(len(emotions))], [e.capitalize() for e in emotions], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')
    elif output_dir:
        plt.savefig(os.path.join(output_dir, "character_emotions.png"))
        plt.close()
        print(f"Character emotions plot saved to {output_dir}")
        return None

def plot_topic_heatmap(profile, output_dir=None, return_base64=False):
    """
    Plot a heatmap of topics by chapter.
    
    Args:
        profile (dict): Book profile data
        output_dir (str): Directory to save plots to
        return_base64 (bool): Whether to return base64 encoded images instead of saving
        
    Returns:
        str: Base64 encoded image if return_base64 is True
    """
    topic_dist = np.array(profile["chapter_topic_distribution"])
    topics = profile["topics"]
    
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Number of chapters and topics
    n_chapters, n_topics = topic_dist.shape
    
    # Create labels for topics
    topic_labels = []
    for i in range(n_topics):
        top_words = topics[f"topic_{i}"][:3]  # Get top 3 words
        topic_labels.append(f"Topic {i}: {', '.join(top_words)}")
    
    # Create the heatmap
    plt.figure(figsize=(18, 10))
    plt.imshow(topic_dist, cmap='YlOrRd')
    plt.colorbar(label='Topic Probability')
    
    # Add labels
    plt.yticks(range(n_chapters), [f"Chapter {i+1}" for i in range(n_chapters)])
    plt.xticks(range(n_topics), topic_labels, rotation=45, ha='right')
    
    plt.title('Topic Distribution Across Chapters')
    plt.tight_layout()
    
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')
    elif output_dir:
        plt.savefig(os.path.join(output_dir, "topic_heatmap.png"))
        plt.close()
        print(f"Topic heatmap saved to {output_dir}")
        return None

def generate_all_visualizations(profile, output_dir=None, return_base64=False):
    """
    Generate all visualizations for a book profile.
    
    Args:
        profile (dict): Book profile data
        output_dir (str): Directory to save plots to
        return_base64 (bool): Whether to return base64 encoded images instead of saving
        
    Returns:
        dict: Base64 encoded images if return_base64 is True
    """
    results = {}
    
    # Generate sentiment trajectory visualizations
    sentiment_vis = plot_sentiment_trajectory(profile, output_dir, return_base64)
    if return_base64 and sentiment_vis:
        results.update(sentiment_vis)
    
    # Generate character emotions visualization
    character_vis = plot_character_emotions(profile, output_dir, return_base64)
    if return_base64 and character_vis:
        results['character_emotions'] = character_vis
    
    # Generate topic heatmap visualization
    topic_vis = plot_topic_heatmap(profile, output_dir, return_base64)
    if return_base64 and topic_vis:
        results['topic_heatmap'] = topic_vis
    
    return results if return_base64 else None 